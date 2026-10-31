#!/usr/bin/env python3
"""Check native-aware CQ primitives against the actual C kernel, CPU only."""
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
    for name in ('checkpoint', 'model', 'binary', 'needle-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve evidence; use a fresh output directory')
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    helper = runpy.run_path(str(Path(__file__).with_name('train_english_student.py')))
    helper['verify_source'](a.needle_source)
    helper['checkpoint_manifest'](a.checkpoint)
    receipt = json.loads((a.model.parent / 'export.json').read_text())
    if sha(a.model) != receipt['sha256'] or sha(a.checkpoint / 'manifest.json') != receipt['checkpoint_manifest_sha256'] or (a.model.parent / 'INCOMPLETE').exists():
        raise ValueError('Changed checkpoint/export')
    _, quantize, checkpoints = helper['source_modules'](a.needle_source)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import english_native_quant as nq
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    params = checkpoints.read_checkpoint(a.checkpoint / 'model.safetensors')['params']
    cases = [(0, 'tied-output', params['embedding']['embedding'], 4),
             (2, 'decoder-query', params['stack']['layers']['block']['self_attn']['q_proj']['kernel'][0].T, 2),
             (61, 'decoder-mhc', params['stack']['mhc_phi_pre'].transpose(0, 2, 1).reshape(8, 2048), 2),
             (452, 'stem-output', params['stem']['out']['kernel'].T, 2)]
    # These directory indices belong to the verified full two-layer research
    # model. Refuse a different geometry instead of mislabelling the test.
    c = receipt['config']
    if (c['text'], c['dec_layers'], c['enc_layers'], c['dim'], c['sites'], receipt['tensors']) != (2048, 2, 8, 512, [1], 455):
        raise ValueError('Unsupported probe geometry')
    a.output.mkdir(parents=True)
    (a.output / 'INCOMPLETE').write_text('Native quantization not yet verified.\n')
    rng = np.random.default_rng(42)
    results = []
    for index, name, weight, bits in cases:
        width = weight.shape[-1]
        x = rng.normal(0, .2, (8, width)).astype(np.float32)
        x[0] = 0; x[1] = 1; x[2] = -1
        source = a.output / (name + '-input.f32')
        target = a.output / (name + '-native.f32')
        x.tofile(source)
        subprocess.run([str(a.binary.resolve()), str(a.model.resolve()), 'cq-linear', str(index), str(source.resolve()), str(target.resolve())], check=True, timeout=60)
        actual = np.fromfile(target, np.float32).reshape(8, weight.shape[0])
        effective = nq.cq_values(jnp.asarray(weight), quantize, bits)
        expected = np.asarray(nq.activation_values(jnp.asarray(x)) @ effective.T)
        error = np.abs(actual - expected)
        record = dict(name=name, tensor_index=index, bits=bits, input_width=width,
                      maximum_absolute_error=float(error.max()),
                      within_tolerance=bool(np.allclose(actual, expected, atol=2e-5, rtol=2e-4)),
                      all_finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all()))
        results.append(record)
        expected.tofile(a.output / (name + '-simulated.f32'))
        print(json.dumps(record), flush=True)
    ties = jnp.asarray([-2.5, -1.5, -.5, 0, .5, 1.5, 2.5], jnp.float32)
    ties_ok = bool(np.array_equal(np.asarray(nq.round_away(ties)), [-3, -2, -1, 0, 1, 2, 3]))
    gradients = jax.grad(lambda x: jnp.sum(nq.activation(x)))(jnp.asarray(x))
    act_gradient_ok = bool(np.array_equal(np.asarray(gradients), np.ones(x.shape, np.float32)))
    probe = jnp.asarray(rng.normal(0, .02, (3, 513)).astype(np.float32))
    cq_gradient = jax.grad(lambda w: jnp.sum(nq.cq_ste(w, quantize, 2)))(probe)
    cq_gradient_ok = bool(np.array_equal(np.asarray(cq_gradient), np.ones(probe.shape, np.float32)))
    fixture = {'embedding': {'embedding': jnp.asarray(rng.normal(0, .02, (7, 128)).astype(np.float32))},
               'engrams_0': {'embedding': jnp.asarray(rng.normal(0, .02, (4, 23, 128)).astype(np.float32)),
                             'key_proj': {'kernel': jnp.asarray(rng.normal(0, .02, (128, 7)).astype(np.float32))}}}
    original = [np.asarray(v).copy() for v in jax.tree.leaves(fixture)]
    converted = nq.parameters(fixture, quantize, 2)
    lookup_ok = (np.array_equal(np.asarray(converted['embedding']['embedding']), np.asarray(quantize.cq_ste(fixture['embedding']['embedding'], 4))) and
                 np.array_equal(np.asarray(converted['engrams_0']['embedding']), np.asarray(quantize.cq_ste(fixture['engrams_0']['embedding'], 2))))
    all_gradients = jax.grad(lambda tree: sum(jnp.sum(v) for v in jax.tree.leaves(nq.parameters(tree, quantize, 2))))(fixture)
    head_grad_ok = bool(np.array_equal(np.asarray(all_gradients['embedding']['embedding']), np.full((7, 128), 2., np.float32)))
    others_grad_ok = all(np.array_equal(np.asarray(v), np.ones(v.shape, np.float32)) for v in jax.tree.leaves(all_gradients['engrams_0']))
    unchanged = all(np.array_equal(before, np.asarray(after)) for before, after in zip(original, jax.tree.leaves(fixture), strict=True))
    table_policy_ok = bool(lookup_ok and head_grad_ok and others_grad_ok and unchanged and nq.OUTPUT_HEAD not in fixture)
    report = dict(schema=1, scope='Four real packed matrices, rounding ties and fixture gradients; not whole-network or speech acceptance',
                  policy=nq.POLICY, model_sha256=sha(a.model), binary_sha256=sha(a.binary),
                  simulation_sha256=sha(Path(nq.__file__)), checker_sha256=sha(Path(__file__)),
                  cases=results, round_half_away_ties_verified=ties_ok,
                  activation_straight_through_gradient_verified=act_gradient_ok,
                  padded_cq_straight_through_gradient_verified=cq_gradient_ok,
                  original_lookup_tied_output_gradient_and_no_mutation_verified=table_policy_ok,
                  absolute_tolerance=2e-5, relative_tolerance=2e-4,
                  all_passed=bool(ties_ok and act_gradient_ok and cq_gradient_ok and table_policy_ok and all(v['within_tolerance'] and v['all_finite'] for v in results)),
                  deployment_permitted=False)
    (a.output / 'quantization.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_passed']:
        raise ValueError('Native quantization primitive comparison failed; preserve evidence')
    (a.output / 'INCOMPLETE').unlink()


if __name__ == '__main__':
    main()
