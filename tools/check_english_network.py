#!/usr/bin/env python3
"""Compare exported C execution with the matching JAX network, CPU only.

This isolates weight packing and architecture. Activation/KV quantization,
speech accuracy and target-device performance require separate acceptance.
Run under an external memory/time guard; this script alone does not limit RSS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import wave


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'model', 'binary', 'needle-source', 'catalog', 'audio-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--quantized', action='store_true',
                   help='Measure native fast CQ execution against trainer A8/KV8 simulation')
    modes.add_argument('--native-aware', action='store_true',
                       help='Evaluate the proposed native-aware simulation on existing weights; does not retrain them')
    p.add_argument('--stage-binary', type=Path,
                   help='Only diagnose encoder stages on the first clip with this host-only probe')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory; preserve previous evidence')
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    helper = runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source)
    manifest = helper['checkpoint_manifest'](a.checkpoint)
    exported = json.loads((a.model.parent / 'export.json').read_text())
    if (a.model.parent / 'INCOMPLETE').exists() or sha(a.model) != exported['sha256']:
        raise ValueError('Incomplete or changed native export')
    if sha(a.checkpoint / 'manifest.json') != exported['checkpoint_manifest_sha256']:
        raise ValueError('Checkpoint does not match export')
    signature = manifest['metadata']['signature']
    if signature != exported['training_signature'] or signature['audio_frontend_profile'] != 3:
        raise ValueError('Expected matching profile-3 training and export')
    import english_features
    import english_precision
    if (sha(Path(english_features.__file__)) != signature['frontend_module_sha256'] or
            sha(Path(english_precision.__file__)) != signature['precision_module_sha256']):
        raise ValueError('Changed training implementation')
    whistle, quantize, checkpoints = helper['source_modules'](a.needle_source)
    english_features.install(whistle)
    if a.quantized:
        quantize.configure_deploy(act_bits=8, kv_bits=8)
    if a.native_aware:
        import english_native_quant
        if signature.get('deploy_policy') == 'native-fast':
            helper['verify_deployment_signature'](signature, english_native_quant)
        english_native_quant.install(whistle, quantize)
        quantize.configure_deploy(act_bits=8, kv_bits=0)
    import jax
    import jax.numpy as jnp
    import numpy as np
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    saved = checkpoints.read_checkpoint(a.checkpoint / 'model.safetensors')
    cfg = whistle.WhistleConfig.from_saved(saved['config'])
    # Export stores non-CQ parameters in FP16. Match those stored values,
    # retaining FP32 computation, to isolate execution rather than rounding.
    deployed = (english_native_quant.parameters(saved['params'], quantize, signature['qat_bits']) if a.native_aware
                else english_precision.qat_params(saved['params'], quantize, signature['qat_bits']))
    def derived_head(path):
        return (a.native_aware and len(path) == 1 and getattr(path[0], 'key', None) == english_native_quant.OUTPUT_HEAD)
    deployed = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf if quantize._is_quant_leaf(path, leaf) or derived_head(path)
        else jnp.asarray(leaf).astype(jnp.float16).astype(jnp.float32), deployed)
    model = whistle.Whistle(cfg)
    tokens = jnp.asarray([[2, cfg.text, 42, 101, 7, 1023, 33, 99]], dtype=jnp.int32)
    quant = a.quantized or a.native_aware
    encode = jax.jit(lambda params, wav, lengths: model.apply({'params': params}, wav, lengths, quant=quant, method=model.encode))
    logits = jax.jit(lambda params, enc, lengths: model.apply({'params': params}, tokens, enc, lengths, quant=quant, method=model.logits))
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Whole-network parity not established.\n')
    cases = []
    root = a.audio_root.resolve()
    for index, item in enumerate(json.loads(a.catalog.read_text())['cases']):
        wav = (root / item['file']).resolve()
        if wav.parent != root or sha(wav) != item['sha256']:
            raise ValueError('Changed natural speech fixture')
        with wave.open(str(wav), 'rb') as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
                raise ValueError('Wrong audio format')
            pcm = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768.
        if a.stage_binary:
            import english_stage_diagnostics
            english_stage_diagnostics.compare(a, model, deployed, cfg, whistle, pcm, quant)
            return
        enc_path = a.output / f'{index}-native-encoder.f32'
        lg_path = a.output / f'{index}-native-logits.f32'
        command = [str(a.binary.resolve()), str(a.model.resolve()), str(wav), '--lang', 'en',
                   '--enc-out', str(enc_path.resolve()), '--logits-out', str(lg_path.resolve()),
                   '--tokens', ','.join(map(str, np.asarray(tokens)[0]))]
        if not quant:
            command.append('--float')
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (a.output / f'{index}-native.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
        actual_enc = np.fromfile(enc_path, np.float32).reshape(-1, cfg.dim)
        actual_logits = np.fromfile(lg_path, np.float32).reshape(tokens.shape[1], cfg.vocab)
        expected_enc, frames = encode(deployed, jnp.asarray(pcm[None, :]), jnp.asarray([len(pcm)]))
        valid = int(frames[0])
        expected_logits = np.asarray(logits(deployed, expected_enc, frames))[0]
        # Feed native encoder memory into JAX to localize decoder differences.
        isolated_logits = np.asarray(logits(deployed, jnp.asarray(actual_enc[None, :, :]), frames))[0]
        expected_enc = np.asarray(expected_enc)[0, :valid]
        if expected_enc.shape != actual_enc.shape:
            raise ValueError(('Encoder shape mismatch', expected_enc.shape, actual_enc.shape))
        expected_enc.tofile(a.output / f'{index}-jax-encoder.f32')
        expected_logits.tofile(a.output / f'{index}-jax-logits.f32')
        isolated_logits.tofile(a.output / f'{index}-jax-native-memory-logits.f32')
        relative = float(np.linalg.norm(actual_enc - expected_enc) / max(float(np.linalg.norm(expected_enc)), 1e-12))
        error = float(np.max(np.abs(actual_logits - expected_logits)))
        isolated_error = float(np.max(np.abs(actual_logits - isolated_logits)))
        finite = all(np.isfinite(v).all() for v in (actual_enc, actual_logits, expected_enc, expected_logits, isolated_logits))
        same = bool(np.array_equal(actual_logits.argmax(-1), expected_logits.argmax(-1)))
        record = dict(id=item['id'], audio_sha256=item['sha256'], frames=valid,
                      encoder_relative_l2_error=relative, logits_max_absolute_error=error,
                      native_memory_logits_max_absolute_error=isolated_error,
                      argmax_identical=same, all_finite=bool(finite),
                      within_tolerance=bool(finite and relative < 2e-4 and error < 2e-3 and isolated_error < 2e-3 and same))
        cases.append(record)
        print(json.dumps(record), flush=True)
    scope = ('Native fast CQ execution versus trainer A8/KV8 simulation; float tolerances retained to expose discrepancies, no accuracy or device acceptance'
             if a.quantized else 'Float execution with matched CQ weights and FP16-rounded other parameters; no A8/KV8, accuracy or device acceptance')
    mode = 'native-fast-vs-trainer-a8-kv8' if a.quantized else 'float'
    if a.native_aware:
        scope = 'Native-aware fast-kernel simulation applied to existing checkpoint; no new training, accuracy or device acceptance'
        mode = english_native_quant.POLICY
    report = dict(schema=1, scope=scope, execution_mode=mode,
                  checkpoint_training_deploy_policy=signature.get('deploy_policy', 'upstream-a8-kv8'),
                  model_sha256=sha(a.model), native_binary_sha256=sha(a.binary), checker_sha256=sha(Path(__file__)),
                  checkpoint_manifest_sha256=sha(a.checkpoint / 'manifest.json'),
                  encoder_relative_tolerance=2e-4, logits_absolute_tolerance=2e-3,
                  tokens=np.asarray(tokens)[0].tolist(), cases=cases,
                  all_passed=all(c['within_tolerance'] for c in cases), deployment_permitted=False)
    if a.native_aware:
        report['simulation_sha256'] = sha(Path(english_native_quant.__file__))
    (a.output / 'network.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_passed']:
        raise ValueError('Whole-network comparison failed; preserve evidence')
    (a.output / 'INCOMPLETE').unlink()


if __name__ == '__main__':
    main()
