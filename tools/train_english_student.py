#!/usr/bin/env python3
"""Bounded full-parameter English Whistle training; no model download or device I/O.

Fresh parameters and a newly trained English tokenizer are mandatory. This
research path does not produce a deployable recognizer or assert speech accuracy.
"""
import argparse
from dataclasses import asdict
from functools import lru_cache, partial
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import random
import runpy
import sys
import time
import types

NEEDLE_COMMIT = 'b4fc11b04f1852888d375c8c6dfbaca44572d014'
NEEDLE_ARCHIVE = '74c157ce6fd599691dd4fd3ae0a2218b03507a9ec3fcbc57adc798767fb49f86'
NEEDLE_RECEIPT = '0217a8bdd09d45d31f2e3ed3db1bfba04fb72c63aa124ec3b35ff98d5836963c'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def verify_source(root):
    if sha(root / 'ACK_SOURCE.json') != NEEDLE_RECEIPT:
        raise ValueError('Training source receipt differs from the pinned archive inventory')
    receipt = json.loads((root / 'ACK_SOURCE.json').read_text())
    if receipt.get('commit') != NEEDLE_COMMIT or receipt.get('archive_sha256') != NEEDLE_ARCHIVE:
        raise ValueError('Training source differs from the pinned source package')
    actual = {str(p.relative_to(root)) for p in (root / 'needle/model').glob('*.py')}
    if actual != set(receipt['files']):
        raise ValueError('Training source file inventory differs')
    for name, expected in receipt['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('Training source file hash differs')
    return sha(root / 'ACK_SOURCE.json')


def source_modules(root):
    # Load the model-only package; do not execute Needle's agent/telemetry facade.
    for name, path in (('_ack_needle', root / 'needle'), ('_ack_needle.model', root / 'needle/model')):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return tuple(importlib.import_module('_ack_needle.model.' + name)
                 for name in ('whistle', 'quantize', 'checkpoints'))


def research_inputs(corpus, tokenizers, vocabulary):
    helpers = runpy.run_path(str(Path(__file__).with_name('train_english_tokenizer.py')))
    splits, receipt = helpers['load_splits'](corpus)
    if (tokenizers / 'INCOMPLETE').exists():
        raise ValueError('Tokenizer job is incomplete')
    catalog = json.loads((tokenizers / 'tokenizers.json').read_text())
    if catalog.get('corpus_sha256') != sha(corpus / 'corpus.json') or catalog.get('validation_or_test_used_for_fitting') is not False:
        raise ValueError('Tokenizer is not tied to the verified training-only corpus')
    candidates = [x for x in catalog['candidates'] if x['vocabulary'] == vocabulary]
    if len(candidates) != 1:
        raise ValueError('Unknown or ambiguous English vocabulary')
    candidate = candidates[0]
    path = (tokenizers / candidate['model']).resolve()
    if path.parent != tokenizers.resolve() or sha(path) != candidate['sha256']:
        raise ValueError('Tokenizer bytes differ from their receipt')
    return splits, path, {'corpus_sha256': sha(corpus / 'corpus.json'),
                          'tokenizer_sha256': candidate['sha256'],
                          'train_manifest_sha256': receipt['manifests']['train']['sha256']}


def eligible_rows(rows, tokenizer, maximum_samples, batch_tokens, limit=0):
    selected, excluded_duration, excluded_tokens = [], 0, 0
    for row in rows:
        if type(row.get('samples')) is not int or row['samples'] <= 0:
            raise ValueError('Invalid audio sample count')
        if row['samples'] > maximum_samples:
            excluded_duration += 1
            continue
        ids = [2, tokenizer.get_piece_size(), *tokenizer.encode(row['text']), 1]
        if len(ids) > batch_tokens + 1:
            excluded_tokens += 1
            continue
        if tokenizer.unk_id() in ids[2:-1]:
            raise ValueError('Unknown transcript piece despite byte fallback')
        selected.append(dict(row, token_ids=ids))
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError('No eligible clips; no audio is truncated automatically')
    return selected, {'eligible': len(selected), 'excluded_duration': excluded_duration,
                      'excluded_tokens': excluded_tokens, 'limit': limit}


def audio_path(row, source_root, local_root):
    # The origin path is in another host's namespace. Resolving it locally can
    # rewrite /srv to /var/srv and break a legitimate remote-prefix mapping.
    source_root, local_root = PurePosixPath(str(source_root)), local_root.resolve()
    source = PurePosixPath(row['audio'])
    if not source.is_absolute() or not source_root.is_absolute() or '..' in source.parts or '..' in source_root.parts:
        raise ValueError('Expected absolute, normalized origin paths')
    relative = source.relative_to(source_root)
    path = (local_root / relative).resolve()
    if not path.is_relative_to(local_root):
        raise ValueError('Audio path escapes the selected cache root')
    return path


@lru_cache(maxsize=2)
def sample_order(count, epoch, seed):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def sample_indices(count, step, batch_size, seed):
    indices = []
    for position in range(step * batch_size, (step + 1) * batch_size):
        epoch, offset = divmod(position, count)
        indices.append(sample_order(count, epoch, seed)[offset])
    return indices


def lazy_batch(rows, indices, source_root, local_root, maximum_samples, batch_tokens):
    import numpy as np
    import soundfile as sf
    wav = np.zeros((len(indices), maximum_samples), np.float32)
    lengths = np.zeros(len(indices), np.int32)
    tokens = np.zeros((len(indices), batch_tokens + 1), np.int32)
    for batch, index in enumerate(indices):
        row = rows[index]
        path = audio_path(row, source_root, local_root)
        if sha(path) != row['audio_sha256']:
            raise ValueError('Selected source audio hash differs')
        samples, rate = sf.read(path, dtype='float32', always_2d=True)
        if rate != 16000 or samples.shape != (row['samples'], 1) or not np.isfinite(samples).all():
            raise ValueError('Source audio differs from verified mono 16k geometry')
        if len(samples) > maximum_samples or len(row['token_ids']) > batch_tokens + 1:
            raise ValueError('Batch would truncate audio or labels')
        wav[batch, :len(samples)] = samples[:, 0]
        lengths[batch] = len(samples)
        tokens[batch, :len(row['token_ids'])] = row['token_ids']
    return wav, lengths, tokens


def checkpoint_manifest(root):
    if (root / 'INCOMPLETE').exists() or not (root / 'READY').is_file():
        raise ValueError('Checkpoint is incomplete')
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['files'].values():
        path = (root / item['file']).resolve()
        if path.parent != root.resolve() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Checkpoint bytes differ from its receipt')
    return manifest


def write_checkpoint(root, params, optimizer_state, metadata, checkpoints, serialization):
    import jax
    root.mkdir(parents=True, exist_ok=False)
    (root / 'INCOMPLETE').write_text('Checkpoint not admitted until manifest and READY exist.\n')
    model_path, optimizer_path = root / 'model.safetensors', root / 'optimizer.msgpack'
    checkpoints.write_checkpoint(model_path, {'params': jax.device_get(params),
        'config': metadata['config'], 'step': metadata['step'], 'run': metadata})
    with optimizer_path.open('xb') as output:
        output.write(serialization.to_bytes(jax.device_get(optimizer_state)))
        output.flush()
        os.fsync(output.fileno())
    with model_path.open('rb') as source:
        os.fsync(source.fileno())
    files = {name: {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for name, path in (('model', model_path), ('optimizer', optimizer_path))}
    with (root / 'manifest.json').open('x') as output:
        output.write(json.dumps({'metadata': metadata, 'files': files}, indent=2) + '\n')
        output.flush()
        os.fsync(output.fileno())
    with (root / 'READY').open('x') as output:
        output.write('Complete research checkpoint; not a deployment artifact.\n')
        output.flush()
        os.fsync(output.fileno())
    (root / 'INCOMPLETE').unlink()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return checkpoint_manifest(root)


def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('corpus', 'tokenizers', 'needle-source', 'audio-source-root', 'audio-local-root', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--vocabulary', type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--batch-tokens', type=int, default=320)
    parser.add_argument('--limit-train-clips', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--qat-bits', type=int, choices=(0, 4), default=4)
    parser.add_argument('--parameter-budget-bytes', type=int, default=256 * 1024 * 1024)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.limit_train_clips < 0 or args.seed < 0 or not 4 <= args.batch_tokens <= 320:
        parser.error('Invalid bounded training count or token budget')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 30 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Invalid duration or learning rate')
    if args.output.exists():
        parser.error('Preserve earlier runs; choose a new output directory')
    # CPU-only until an explicit GPU reservation/runner is implemented.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'INCOMPLETE').write_text('Research training run not complete.\n')
    started = time.monotonic()
    def progress(stage, **values):
        record = dict(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3), **values)
        with (args.output / 'events.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        print(stage, json.dumps(values), flush=True)
    progress('verify_inputs')
    source_hash = verify_source(args.needle_source)
    splits, tokenizer_path, inputs = research_inputs(args.corpus, args.tokenizers, args.vocabulary)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import sentencepiece as spm
    from flax import serialization
    from flax.core import unfreeze
    if jax.default_backend() != 'cpu':
        raise ValueError('This bounded runner must not acquire a GPU')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if (tokenizer.get_piece_size(), tokenizer.pad_id(), tokenizer.eos_id(), tokenizer.bos_id(), tokenizer.unk_id()) != (args.vocabulary, 0, 1, 2, 3):
        raise ValueError('English tokenizer special IDs differ')
    rows, selection = eligible_rows(splits['train'], tokenizer, int(args.max_seconds * 16000), args.batch_tokens, args.limit_train_clips)
    ids_hash = hashlib.sha256(canonical([row['id'] for row in rows]).encode()).hexdigest()
    whistle, quantize, checkpoints = source_modules(args.needle_source)
    cfg = whistle.WhistleConfig(text=args.vocabulary, languages=('en',), dim=512,
        enc_layers=8, dec_layers=2, sites=(1,), slots=18432, flash=False, remat=True)
    model = whistle.Whistle(cfg)
    maximum_samples = int(args.max_seconds * 16000)
    first = lazy_batch(rows, sample_indices(len(rows), 0, args.batch_size, args.seed),
        args.audio_source_root, args.audio_local_root, maximum_samples, args.batch_tokens)
    shape = unfreeze(jax.eval_shape(lambda: model.init(jax.random.PRNGKey(args.seed),
        jnp.asarray(first[0]), jnp.asarray(first[1]), jnp.asarray(first[2][:, :-1]), quant=False))['params'])
    count = sum(math.prod(x.shape) for x in jax.tree.leaves(shape))
    progress('parameter_plan', parameters=count, parameter_bytes=count * 4, languages=['en'])
    if count * 4 > args.parameter_budget_bytes:
        raise ValueError('Full target parameter tree exceeds the declared allocation budget')
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr))
    versions = {name: importlib.metadata.version(name) for name in
                ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'safetensors', 'sentencepiece', 'soundfile')}
    signature = {**inputs, 'source_receipt_sha256': source_hash, 'config': asdict(cfg),
        'training_script_sha256': sha(Path(__file__)), 'packages': versions,
        'python': sys.version.split()[0], 'sampler': 'Random(seed+epoch) shuffle with global-step offset',
        'seed': args.seed, 'batch_size': args.batch_size, 'maximum_samples': maximum_samples,
        'batch_tokens': args.batch_tokens, 'qat_bits': args.qat_bits, 'lr': args.lr,
        'selected_train_ids_sha256': ids_hash, 'optimizer': 'clip1.0 + optax.adamw(default weight_decay)'}
    with (args.output / 'plan.json').open('x') as output:
        output.write(json.dumps({'signature': signature, 'parameters': count,
                                 'parameter_bytes': count * 4, 'selection': selection}, indent=2) + '\n')
    if args.resume:
        manifest = checkpoint_manifest(args.resume)
        if canonical(manifest['metadata']['signature']) != canonical(signature):
            raise ValueError('Resume inputs, geometry or optimizer differ')
        saved = checkpoints.read_checkpoint(args.resume / 'model.safetensors')
        params = saved['params']
        if jax.tree.structure(shape) != jax.tree.structure(params):
            raise ValueError('Checkpoint parameter tree differs')
        for expected, value in zip(jax.tree.leaves(shape), jax.tree.leaves(params), strict=True):
            if expected.shape != value.shape or expected.dtype != value.dtype:
                raise ValueError('Checkpoint parameter shape or dtype differs')
        template = jax.eval_shape(optimizer.init, shape)
        opt_state = serialization.from_bytes(template, (args.resume / 'optimizer.msgpack').read_bytes())
        if jax.tree.structure(template) != jax.tree.structure(opt_state):
            raise ValueError('Checkpoint optimizer tree differs')
        for expected, value in zip(jax.tree.leaves(template), jax.tree.leaves(opt_state), strict=True):
            if expected.shape != value.shape or expected.dtype != value.dtype:
                raise ValueError('Checkpoint optimizer shape or dtype differs')
        step = manifest['metadata']['step']
        progress('restored_checkpoint', step=step)
        params, opt_state = jax.device_put(params), jax.device_put(opt_state)
    else:
        progress('initialize_fresh_full_model')
        params = unfreeze(model.init(jax.random.PRNGKey(args.seed), jnp.asarray(first[0]),
            jnp.asarray(first[1]), jnp.asarray(first[2][:, :-1]), quant=False)['params'])
        opt_state, step = optimizer.init(params), 0
    quantize.configure_deploy(act_bits=8, kv_bits=8)
    # Initialization may share constant-zero buffers among norm leaves. Give
    # each donated leaf its own buffer; this does not change parameter values.
    params, opt_state = jax.tree.map(lambda value: jnp.array(value, copy=True), (params, opt_state))
    def loss_fn(parameters, wav, lengths, tokens):
        deployed = quantize.cq_ste_params(parameters, args.qat_bits) if args.qat_bits else parameters
        logits = model.apply({'params': deployed}, wav, lengths, tokens[:, :-1], quant=bool(args.qat_bits))
        mask = tokens[:, 1:] != cfg.pad
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
        return (loss * mask).sum() / jnp.maximum(mask.sum(), 1)
    @partial(jax.jit, donate_argnums=(0, 1))
    def training_step(parameters, state, wav, lengths, tokens):
        loss, gradients = jax.value_and_grad(loss_fn)(parameters, wav, lengths, tokens)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradients)]))
        updates, state = optimizer.update(gradients, state, parameters)
        updated = optax.apply_updates(parameters, updates)
        return updated, state, loss, optax.global_norm(gradients), optax.global_norm(updates), finite
    records, initial_step = [], step
    for _ in range(args.steps):
        batch = lazy_batch(rows, sample_indices(len(rows), step, args.batch_size, args.seed),
            args.audio_source_root, args.audio_local_root, maximum_samples, args.batch_tokens)
        progress('training_step', step=step + 1)
        params, opt_state, loss, norm, update, finite = training_step(params, opt_state, *map(jnp.asarray, batch))
        record = {'step': step + 1, 'loss': float(loss), 'gradient_norm': float(norm), 'update_norm': float(update), 'all_gradients_finite': bool(finite)}
        if not record['all_gradients_finite'] or not all(math.isfinite(record[x]) for x in ('loss', 'gradient_norm', 'update_norm')) or record['update_norm'] <= 0:
            raise ValueError('Nonfinite or empty full-parameter update')
        step += 1
        records.append(record)
        progress('step_complete', **record)
    metadata = {'config': asdict(cfg), 'step': step, 'signature': signature,
        'initialization': 'fresh random, no inherited model weights', 'all_parameters_trainable': True,
        'trainable_parameters': count, 'neural_accuracy_verified': False, 'deployment_permitted': False}
    progress('write_checkpoint', step=step)
    checkpoint = args.output / ('step-' + str(step).zfill(8))
    manifest = write_checkpoint(checkpoint, params, opt_state, metadata, checkpoints, serialization)
    report = {'schema': 1, 'scope': 'Bounded full-parameter CPU training check; not an accepted speech model',
        'source_commit': NEEDLE_COMMIT, 'signature': signature, 'parameters': count,
        'all_parameters_trainable': True, 'frozen_parameters': 0, 'initial_step': initial_step,
        'final_step': step, 'resumed': bool(args.resume), 'selection': selection, 'steps': records,
        'checkpoint': checkpoint.name, 'checkpoint_manifest_sha256': sha(checkpoint / 'manifest.json'),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'backend': jax.default_backend(), 'neural_accuracy_verified': False,
        'native_export_compatibility_verified': False, 'deployment_permitted': False}
    with (args.output / 'training.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    (args.output / 'INCOMPLETE').unlink()
    progress('complete', final_step=step)


if __name__ == '__main__':
    main()
