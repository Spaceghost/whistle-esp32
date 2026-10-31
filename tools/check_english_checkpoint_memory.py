#!/usr/bin/env python3
"""Measure CPU optimizer serialization with real model shapes and synthetic moments.

Run each writer in its own guarded process. This is not training or recovery of
an incomplete checkpoint. Compare the two output hashes before claiming parity.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import train_english_student as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--needle-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('legacy', 'stream'), required=True)
    parser.add_argument('--parameters', type=int, default=29537627)
    args = parser.parse_args()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'INCOMPLETE').write_text('Serialization measurement unfinished.\n')
    source = core.verify_source(args.needle_source)
    import jax
    import numpy as np
    import optax
    from flax import serialization
    import english_checkpoint_stream as stream
    import english_training_memory as memory
    if jax.default_backend() != 'cpu':
        raise ValueError('Serialization measurement requires CPU execution')
    _, _, checkpoints = core.source_modules(args.needle_source)
    params = checkpoints.read_checkpoint(args.model)['params']
    leaves = jax.tree.leaves(params)
    count = sum(value.size for value in leaves)
    if count != args.parameters or any(value.dtype != np.dtype('float32') for value in leaves):
        raise ValueError('Model geometry or dtype differs from the expected full student')
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4))
    state = optimizer.init(params)
    jax.block_until_ready(state)
    digest = memory.state_digest(jax, state)
    before = memory.rss_bytes()
    started = time.monotonic()
    output = args.output/'optimizer.msgpack'
    with output.open('xb') as file:
        if args.mode == 'stream':
            stats = stream.stream_optimizer(file, state, serialization)
        else:
            payload = serialization.to_bytes(jax.device_get(state))
            file.write(payload)
            stats = dict(format='flax-to-bytes', bytes=len(payload))
            del payload
        file.flush()
        os.fsync(file.fileno())
    elapsed = time.monotonic()-started
    after = memory.rss_bytes()
    if memory.state_digest(jax, state) != digest:
        raise ValueError('Synthetic optimizer state changed during serialization')
    report = dict(schema=1, scope='CPU serialization of full model geometry with synthetic initial Adam moments; not trained-state recovery',
        mode=args.mode, parameters=count, optimizer_bytes=digest[1], optimizer_leaves=digest[2],
        optimizer_state_sha256=digest[0], state_bit_identical=True,
        model_sha256=core.sha(args.model), source_receipt_sha256=source,
        writer_sha256=core.sha(Path(stream.__file__)), file_bytes=output.stat().st_size,
        file_sha256=core.sha(output), baseline_rss_bytes=before, rss_after_write_bytes=after,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        write_seconds=elapsed, writer=stats, training_performed=False, deployment_permitted=False)
    (args.output/'memory.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'INCOMPLETE').unlink()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
