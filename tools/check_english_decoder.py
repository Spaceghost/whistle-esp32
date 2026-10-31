#!/usr/bin/env python3
"""Localize decoder/head differences using retained native encoder memory.

Run under guard_english_job.py. This is diagnosis, never deployment acceptance.
The selected network receipt must match the checkpoint, model and execution mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'model', 'needle-source', 'network', 'binary', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--case-index', type=int, default=3)
    p.add_argument('--float', dest='floating', action='store_true')
    p.add_argument('--layers', action='store_true', help='Compare native and simulated decoder layer boundaries')
    p.add_argument('--sublayers', action='store_true', help='Also compare first-block self/cross-attention and feed-forward boundaries')
    p.add_argument('--cross-attention', action='store_true', help='Compare first-block cross-attention projections, cache and Q8 decisions')
    p.add_argument('--cross-composition', action='store_true', help='Compare composed normalization/attention and a diagnostic optimization barrier')
    p.add_argument('--linear-binary', type=Path, help='Native profile_probe used for controlled query projection replay')
    a = p.parse_args()
    if a.cross_composition:
        a.cross_attention = True
    if a.cross_attention:
        a.sublayers = True
        if a.linear_binary is None or not a.linear_binary.is_file():
            p.error('--cross-attention requires a built --linear-binary profile_probe')
    if a.sublayers:
        a.layers = True
    if a.output.exists():
        p.error('Preserve evidence; choose a fresh output directory')
    if not a.binary.is_file() or not os.access(a.binary, os.X_OK):
        p.error('Build the host decoder probe before loading research weights')
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    helper = runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source)
    manifest = helper['checkpoint_manifest'](a.checkpoint)
    signature = manifest['metadata']['signature']
    export = json.loads((a.model.parent / 'export.json').read_text())
    network = json.loads((a.network / 'network.json').read_text())
    for record in (export, network):
        if record.get('model_sha256', record.get('sha256')) != sha(a.model) or record['checkpoint_manifest_sha256'] != sha(a.checkpoint / 'manifest.json'):
            raise ValueError('Mismatched checkpoint/model/network evidence')
    if (a.model.parent / 'INCOMPLETE').exists() or signature != export['training_signature']:
        raise ValueError('Incomplete export or changed signature')
    import english_features
    import english_precision
    import english_native_quant
    if sha(Path(english_features.__file__)) != signature['frontend_module_sha256'] or sha(Path(english_precision.__file__)) != signature['precision_module_sha256']:
        raise ValueError('Changed training implementation')
    helper['verify_deployment_signature'](signature, english_native_quant)
    mode = 'float' if a.floating else english_native_quant.POLICY
    if network['execution_mode'] != mode or not 0 <= a.case_index < len(network['cases']):
        raise ValueError('Wrong execution mode or clip index')
    whistle, quantize, checkpoints = helper['source_modules'](a.needle_source)
    english_features.install(whistle)
    if not a.floating:
        english_native_quant.install(whistle, quantize)
        quantize.configure_deploy(act_bits=8, kv_bits=0)
    import jax
    import jax.numpy as jnp
    import numpy as np
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    saved = checkpoints.read_checkpoint(a.checkpoint / 'model.safetensors')
    import english_runtime
    cfg = english_runtime.configuration(whistle, saved['config'], signature)
    params = (english_precision.qat_params(saved['params'], quantize, signature['qat_bits']) if a.floating else
              english_native_quant.parameters(saved['params'], quantize, signature['qat_bits']))
    def stored(path, leaf):
        derived = len(path) == 1 and getattr(path[0], 'key', None) == english_native_quant.OUTPUT_HEAD
        return leaf if quantize._is_quant_leaf(path, leaf) or derived else jnp.asarray(leaf).astype(jnp.float16).astype(jnp.float32)
    params = jax.tree_util.tree_map_with_path(stored, params)
    case = network['cases'][a.case_index]
    enc_path = a.network / f'{a.case_index}-native-encoder.f32'
    reference_path = a.network / f'{a.case_index}-native-logits.f32'
    enc = np.fromfile(enc_path, np.float32).reshape(1, case['frames'], cfg.dim)
    tokens = np.asarray(network['tokens'], dtype='<u4')
    reference = np.fromfile(reference_path, np.float32).reshape(len(tokens), cfg.vocab)
    if not np.isfinite(enc).all() or not np.isfinite(reference).all() or np.any(tokens >= cfg.vocab):
        raise ValueError('Invalid retained input')
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Decoder diagnosis incomplete.\n')
    token_path = a.output / 'tokens.u32'
    tokens.tofile(token_path)
    command = [str(a.binary.resolve()), str(a.model.resolve()), str(enc_path.resolve()), str(token_path.resolve()), str(a.output.resolve())]
    if a.floating:
        command.append('--float')
    subprocess.run(command, check=True, timeout=120)
    actual_hidden = np.fromfile(a.output / 'hidden.f32', np.float32).reshape(1, len(tokens), cfg.dim)
    actual_logits = np.fromfile(a.output / 'logits.f32', np.float32).reshape(len(tokens), cfg.vocab)
    if not np.array_equal(actual_logits.view(np.uint32), reference.view(np.uint32)):
        raise ValueError('Probe fails to reproduce retained native logits bit for bit')
    model = whistle.Whistle(cfg)
    decode = jax.jit(lambda v: model.apply({'params': params}, jnp.asarray(tokens[None, :], dtype=jnp.int32), v,
                                        jnp.asarray([case['frames']]), quant=not a.floating, method=model.decode))
    expected_hidden = np.asarray(decode(jnp.asarray(enc)))
    head = params['embedding']['embedding'] if a.floating else params[english_native_quant.OUTPUT_HEAD]
    project = jax.jit(lambda h: (h if a.floating else english_native_quant.activation(h)) @ head.T)
    expected_logits = np.asarray(project(jnp.asarray(expected_hidden)))[0]
    shared_logits = np.asarray(project(jnp.asarray(actual_hidden)))[0]
    expected_hidden.tofile(a.output / 'jax-hidden.f32')
    expected_logits.tofile(a.output / 'jax-logits.f32')
    shared_logits.tofile(a.output / 'jax-native-hidden-logits.f32')
    rows = []
    for i in range(len(tokens)):
        h, target = actual_hidden[0, i], expected_hidden[0, i]
        if not all(np.isfinite(v).all() for v in (h, target, actual_logits[i], expected_logits[i], shared_logits[i])):
            raise ValueError('Nonfinite decoder output')
        rows.append(dict(position=i, input_token=int(tokens[i]),
                         hidden_relative_l2_error=float(np.linalg.norm(h-target) / max(float(np.linalg.norm(target)), 1e-12)),
                         logits_max_absolute_error=float(np.max(np.abs(actual_logits[i]-expected_logits[i]))),
                         shared_hidden_head_max_absolute_error=float(np.max(np.abs(actual_logits[i]-shared_logits[i])))))
    report = dict(schema=1, scope=__doc__, case=case['id'], audio_sha256=case['audio_sha256'], execution_mode=mode,
                  training_runtime_settings=english_runtime.describe(cfg),
                  runtime_configuration_module_sha256=sha(Path(english_runtime.__file__)),
                  model_sha256=sha(a.model), checkpoint_manifest_sha256=sha(a.checkpoint / 'manifest.json'),
                  network_receipt_sha256=sha(a.network / 'network.json'), encoder_input_sha256=sha(enc_path),
                  native_binary_sha256=sha(a.binary), checker_sha256=sha(Path(__file__)),
                  native_probe_logits_bit_identical=True, rows=rows,
                  thresholds_unchanged=True, deployment_permitted=False)
    if a.layers:
        import english_decoder_layers
        report['layers'] = english_decoder_layers.compare(a, model, params, cfg, whistle, tokens, enc, expected_hidden)
        report['layer_checker_sha256'] = sha(Path(english_decoder_layers.__file__))
        if a.sublayers:
            import english_decoder_sublayers
            report['sublayer_checker_sha256'] = sha(Path(english_decoder_sublayers.__file__))
            provenance = a.binary.parent / 'decoder-trace-source.json'
            report['native_trace_source'] = json.loads(provenance.read_text())
        if a.cross_attention:
            import english_cross_attention
            report['cross_checker_sha256'] = sha(Path(english_cross_attention.__file__))
        if a.cross_composition:
            import english_cross_composition
            report['composition_checker_sha256'] = sha(Path(english_cross_composition.__file__))
    (a.output / 'decoder.json').write_text(json.dumps(report, indent=2)+'\n')
    if a.layers and not report['layers']['original_jax_control_passed']:
        raise ValueError('Layer trace changed the JAX baseline; preserve evidence, do not attribute it to native execution')
    if a.sublayers and not report['layers']['sublayers']['original_module_control_passed']:
        raise ValueError('Sublayer trace changed the original block; preserve evidence')
    if a.cross_attention and not report['layers']['cross_attention']['original_module_control_passed']:
        raise ValueError('Cross-attention trace changed the original module; preserve evidence')
    if a.cross_composition and not report['layers']['cross_attention']['composition']['original_module_control_passed']:
        raise ValueError('Composed trace changed the original normalization/attention module; preserve evidence')
    (a.output / 'INCOMPLETE').unlink()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
