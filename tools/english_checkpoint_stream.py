"""Write Flax-compatible optimizer bytes without copying the whole state to RAM."""
import json
import os


def stream_optimizer(output, state, serialization):
    import msgpack
    tree = serialization.to_state_dict(state)
    packer = msgpack.Packer(strict_types=True)
    stats = dict(format='flax-msgpack-stream-v1', bytes=0, leaves=0,
                 largest_encoded_leaf_bytes=0, array_bytes=0)

    def emit(data):
        if output.write(data) != len(data):
            raise OSError('Short optimizer checkpoint write')
        stats['bytes'] += len(data)

    def visit(value):
        if isinstance(value, dict):
            emit(packer.pack_map_header(len(value)))
            for key, child in value.items():
                emit(serialization.msgpack_serialize(key, in_place=True))
                visit(child)
        else:
            # Flax owns array/scalar/chunk encoding. Only this leaf becomes a
            # host array/encoded buffer; the rest stays on its original device.
            encoded = serialization.msgpack_serialize(value, in_place=False)
            emit(encoded)
            stats['leaves'] += 1
            stats['largest_encoded_leaf_bytes'] = max(stats['largest_encoded_leaf_bytes'], len(encoded))
            stats['array_bytes'] += int(getattr(value, 'nbytes', 0))

    visit(tree)
    return stats


def write_checkpoint(root, params, optimizer_state, metadata, checkpoints, serialization):
    import jax
    from train_english_student import checkpoint_manifest, sha
    root.mkdir(parents=True, exist_ok=False)
    (root / 'INCOMPLETE').write_text('Checkpoint not admitted until manifest and READY exist.\n')
    model_path, optimizer_path = root / 'model.safetensors', root / 'optimizer.msgpack'
    checkpoints.write_checkpoint(model_path, {'params': jax.device_get(params),
        'config': metadata['config'], 'step': metadata['step'], 'run': metadata})
    with model_path.open('rb') as source:
        os.fsync(source.fileno())
    with optimizer_path.open('xb') as output:
        stats = stream_optimizer(output, optimizer_state, serialization)
        output.flush()
        os.fsync(output.fileno())
    files = {name: {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for name, path in (('model', model_path), ('optimizer', optimizer_path))}
    with (root / 'manifest.json').open('x') as output:
        output.write(json.dumps({'metadata': metadata, 'files': files, 'optimizer_io': stats}, indent=2) + '\n')
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
