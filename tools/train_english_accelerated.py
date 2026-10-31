#!/usr/bin/env python3
"""Accelerated English student with periodic durable checkpoints.

CPU is the default. CUDA requires current guard admission and a successful,
matching backend receipt. The original CPU trainer stays byte-for-byte intact.
"""
import argparse
from dataclasses import asdict
from functools import partial
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import train_english_student as core
from train_english_student import (NEEDLE_COMMIT, sha, canonical, verify_source,
    research_inputs, eligible_rows, source_modules, lazy_batch, sample_indices,
    deployment_signature, checkpoint_manifest, write_checkpoint)
import english_training_control as control

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
    parser.add_argument('--qat-bits', type=int, choices=(0, 2, 4), default=2)
    parser.add_argument('--deploy-policy', choices=('upstream-a8-kv8', 'native-fast'), default='upstream-a8-kv8',
                        help='Experimental native-fast matches this fork kernel; creates an incompatible new training signature')
    parser.add_argument('--parameter-budget-bytes', type=int, default=256 * 1024 * 1024)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--backend', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--gpu-backend-receipt',type=Path)
    parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--checkpoint-reserve-gib',type=float,default=2)
    args = parser.parse_args()
    if args.deploy_policy == 'native-fast' and not args.qat_bits:
        parser.error('Native-fast training requires two- or four-bit CQ weights')
    if args.steps < 1 or args.batch_size < 1 or args.limit_train_clips < 0 or args.seed < 0 or not 4 <= args.batch_tokens <= 320 or args.checkpoint_every<1:
        parser.error('Invalid bounded training count or token budget')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 30 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Invalid duration or learning rate')
    if args.output.exists():
        parser.error('Preserve earlier runs; choose a new output directory')
    if not math.isfinite(args.checkpoint_reserve_gib) or args.checkpoint_reserve_gib<0:
        parser.error('Invalid checkpoint disk reserve')
    execution,execution_evidence=control.prepare_execution(args.backend,args.gpu_backend_receipt)
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
    if jax.default_backend() != ('gpu' if args.backend=='cuda' else 'cpu'):
        raise ValueError('Selected execution backend is unavailable; no fallback')
    jax.config.update('jax_default_matmul_precision','highest')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if (tokenizer.get_piece_size(), tokenizer.pad_id(), tokenizer.eos_id(), tokenizer.bos_id(), tokenizer.unk_id()) != (args.vocabulary, 0, 1, 2, 3):
        raise ValueError('English tokenizer special IDs differ')
    rows, selection = eligible_rows(splits['train'], tokenizer, int(args.max_seconds * 16000), args.batch_tokens, args.limit_train_clips)
    ids_hash = hashlib.sha256(canonical([row['id'] for row in rows]).encode()).hexdigest()
    whistle, quantize, checkpoints = source_modules(args.needle_source)
    import english_features
    import english_precision
    english_features.install(whistle)
    if args.deploy_policy == 'native-fast':
        import english_native_quant
        english_native_quant.install(whistle, quantize)
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
        'tied_embedding_bits': english_precision.TIED_EMBEDDING_BITS,
        'audio_frontend_profile': english_features.PROFILE,
        'frontend_module_sha256': sha(Path(english_features.__file__)),
        'precision_module_sha256': sha(Path(english_precision.__file__)),
        'selected_train_ids_sha256': ids_hash, 'optimizer': 'clip1.0 + optax.adamw(default weight_decay)'}
    signature.update(deployment_signature(args.deploy_policy,
                     english_native_quant if args.deploy_policy == 'native-fast' else None))
    signature.update(execution=execution,training_helper_sha256=sha(Path(core.__file__)),
                     training_control_sha256=sha(Path(control.__file__)))
    with (args.output / 'plan.json').open('x') as output:
        output.write(json.dumps({'signature': signature, 'parameters': count,
                                 'parameter_bytes': count * 4, 'selection': selection,
                                 'execution_evidence':execution_evidence,'checkpoint_every':args.checkpoint_every}, indent=2) + '\n')
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
    quantize.configure_deploy(act_bits=8, kv_bits=0 if args.deploy_policy == 'native-fast' else 8)
    # Initialization may share constant-zero buffers among norm leaves. Give
    # each donated leaf its own buffer; this does not change parameter values.
    params, opt_state = jax.tree.map(lambda value: jnp.array(value, copy=True), (params, opt_state))
    def loss_fn(parameters, wav, lengths, tokens):
        if args.deploy_policy == 'native-fast':
            deployed = english_native_quant.parameters(parameters, quantize, args.qat_bits)
        else:
            deployed = english_precision.qat_params(parameters, quantize, args.qat_bits) if args.qat_bits else parameters
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
    records,initial_step,saved = [],step,[]
    checkpoint=manifest=None;last_saved=None
    def save(final=False):
        nonlocal checkpoint,manifest,last_saved
        if not control.checkpoint_due(step,args.checkpoint_every,last_saved,final):
            return
        # Params + Adam moments, format overhead and the requested spare disk.
        if shutil.disk_usage(args.output).free<count*12+16*1024**2+args.checkpoint_reserve_gib*1024**3:
            raise ValueError('Insufficient checkpoint space; existing checkpoints preserved')
        metadata={'config':asdict(cfg),'step':step,'signature':signature,
            'initialization':'fresh random, no inherited model weights','all_parameters_trainable':True,
            'trainable_parameters':count,'neural_accuracy_verified':False,'deployment_permitted':False}
        checkpoint=args.output/('step-'+str(step).zfill(8))
        progress('write_checkpoint',step=step)
        manifest=write_checkpoint(checkpoint,params,opt_state,metadata,checkpoints,serialization)
        last_saved=step
        saved.append(dict(step=step,checkpoint=checkpoint.name,manifest_sha256=sha(checkpoint/'manifest.json')))
        progress('checkpoint_ready',step=step,checkpoint=checkpoint.name)
    with control.StopRequest() as stop:
        for _ in range(args.steps):
            if stop.requested:
                break
            batch=lazy_batch(rows,sample_indices(len(rows),step,args.batch_size,args.seed),
                args.audio_source_root,args.audio_local_root,maximum_samples,args.batch_tokens)
            progress('training_step',step=step+1)
            params,opt_state,loss,norm,update,finite=training_step(params,opt_state,*map(jnp.asarray,batch))
            record={'step':step+1,'loss':float(loss),'gradient_norm':float(norm),'update_norm':float(update),'all_gradients_finite':bool(finite)}
            if not record['all_gradients_finite'] or not all(math.isfinite(record[x]) for x in ('loss','gradient_norm','update_norm')) or record['update_norm']<=0:
                raise ValueError('Nonfinite or empty full-parameter update')
            step+=1;records.append(record);progress('step_complete',**record)
            save()
        save(final=True)
        interrupted=stop.requested
    report = {'schema': 1, 'scope': 'Bounded accelerated full-parameter run; not an accepted speech model',
        'source_commit': NEEDLE_COMMIT, 'signature': signature, 'parameters': count,
        'all_parameters_trainable': True, 'frozen_parameters': 0, 'initial_step': initial_step,
        'final_step': step, 'resumed': bool(args.resume), 'selection': selection, 'steps': records,
        'checkpoint': checkpoint.name, 'checkpoint_manifest_sha256': sha(checkpoint / 'manifest.json'),
        'checkpoints':saved,'checkpoint_every':args.checkpoint_every,
        'stop_requested':interrupted,'requested_steps_completed':step-initial_step==args.steps,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'backend': jax.default_backend(), 'neural_accuracy_verified': False,
        'native_export_compatibility_verified': False, 'deployment_permitted': False}
    with (args.output / 'training.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    (args.output / 'INCOMPLETE').unlink()
    progress('stopped_at_checkpoint' if interrupted else 'complete', final_step=step)


if __name__ == '__main__':
    main()
