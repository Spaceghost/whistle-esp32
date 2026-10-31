#!/usr/bin/env python3
"""Measure native block sensitivity to tiny controlled input perturbations.

This is a diagnosis, not a relaxed acceptance test. Original thresholds and
failed comparisons stay intact. No training or physical-device claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'binary', 'stages', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve evidence; choose a fresh output directory')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import numpy as np
    stages = json.loads((a.stages / 'stages.json').read_text())
    if sha(a.model) != stages['model_sha256']:
        raise ValueError('Changed model')
    source = a.stages / 'native-stages' / 'lanes-6.f32'
    retained_output = a.stages / 'native-stages' / 'lanes-7.f32'
    simulated_output = a.stages / 'isolated-lanes-7-jax.f32'
    x = np.fromfile(source, np.float32)
    retained = np.fromfile(retained_output, np.float32)
    simulated = np.fromfile(simulated_output, np.float32)
    if not x.size or x.shape != retained.shape or x.shape != simulated.shape or not all(np.isfinite(v).all() for v in (x, retained, simulated)):
        raise ValueError('Wrong stage geometry or nonfinite input')
    a.output.mkdir()
    scale = np.float32(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    direction = np.random.default_rng(0).normal(size=x.shape).astype(np.float32)
    direction /= np.float32(np.sqrt(np.mean(direction.astype(np.float64) ** 2)))
    variants = [('baseline', x), ('repeat', x),
                ('ulp-up', np.nextafter(x, np.float32(np.inf))),
                ('ulp-down', np.nextafter(x, np.float32(-np.inf)))]
    for exponent in (-7, -6, -5):
        for sign in (-1, 1):
            variants.append((f'rms-1e{exponent}-sign{sign}', x + np.float32(sign * 10. ** exponent) * scale * direction))
    def relative(v, reference):
        return float(np.linalg.norm(v.astype(np.float64) - reference) / max(float(np.linalg.norm(reference.astype(np.float64))), 1e-12))
    results = []
    baseline_identity = None
    for mode in ('fast', 'float'):
        baseline = None
        for name, values in variants:
            target = a.output / (mode + '-' + name)
            target.mkdir()
            audio = target / 'input.f32'
            values.tofile(audio)
            command = [str(a.binary.resolve()), str(a.model.resolve()), str(audio.resolve()), str(target.resolve()), '--block', '7']
            if mode == 'float':
                command.append('--float')
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            out = np.fromfile(target / 'block.f32', np.float32)
            if out.shape != retained.shape or not np.isfinite(out).all():
                raise ValueError('Wrong output geometry or nonfinite output')
            if baseline is None:
                baseline = out
                if mode == 'fast':
                    baseline_identity = bool(np.array_equal(out.view(np.uint32), retained.view(np.uint32)))
            record = dict(mode=mode, variant=name, input_relative_l2_change=relative(values, x),
                          output_relative_l2_change=relative(out, baseline), maximum_absolute_output_change=float(np.max(np.abs(out - baseline))))
            if name == 'repeat' and not np.array_equal(out.view(np.uint32), baseline.view(np.uint32)):
                raise ValueError('Native replay is not deterministic')
            results.append(record)
            print(json.dumps(record), flush=True)
    report = dict(schema=1, scope=__doc__, model_sha256=sha(a.model), binary_sha256=sha(a.binary),
                  input_sha256=sha(source), retained_output_sha256=sha(retained_output), simulation_output_sha256=sha(simulated_output),
                  checker_sha256=sha(Path(__file__)), cases=results, baseline_bit_identical_to_retained=baseline_identity,
                  retained_simulation_relative_l2_error=relative(simulated, retained), acceptance_threshold_unchanged=2e-4,
                  native_fast_numeric_gate_passed=False, training_performed=False, deployment_permitted=False)
    (a.output / 'sensitivity.json').write_text(json.dumps(report, indent=2) + '\n')
    if not baseline_identity:
        raise ValueError('Replay differs from retained baseline; do not attribute change solely to input noise')


if __name__ == '__main__':
    main()
