"""Real Flax byte compatibility, Adam recovery and interrupted checkpoint safety."""
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import serialization
    from safetensors.numpy import save_file
except ImportError:
    jax = None
import english_checkpoint_stream as stream
import train_english_student as core


class ModelFiles:
    @staticmethod
    def write_checkpoint(path, data):
        save_file({'w': np.asarray(data['params']['w'])}, str(path),
                  metadata={'step': str(data['step'])})


@unittest.skipIf(jax is None, 'Run checkpoint controls in the locked training environment')
class StreamingCheckpointTests(unittest.TestCase):
    def state(self):
        params = {'w': jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64}
        optimizer = optax.adamw(1e-4)
        return params, optimizer.init(params), optimizer

    def metadata(self):
        return dict(config={}, step=1, signature={'test': 'synthetic'})

    def test_optimizer_bytes_match_flax_exactly_including_array_chunks(self):
        state = {'weight': np.arange(256, dtype=np.float32).reshape(16, 16),
                 'nested': {'count': np.array(2, dtype=np.int32),
                            'empty': np.empty((0, 3), dtype=np.float16)},
                 'label': 'synthetic', 'raw': b'\x00\xff', 'complex': 1+2j}
        for limit in (serialization.MAX_CHUNK_SIZE, 64):
            with self.subTest(chunk_limit=limit), patch.object(serialization, 'MAX_CHUNK_SIZE', limit):
                expected = serialization.to_bytes(state)
                output = io.BytesIO()
                stats = stream.stream_optimizer(output, state, serialization)
                self.assertEqual(output.getvalue(), expected)
                self.assertEqual(stats['bytes'], len(expected))
                self.assertLess(stats['largest_encoded_leaf_bytes'], len(expected))

    def test_real_adam_checkpoint_restores_bit_identical_next_update(self):
        params, state, optimizer = self.state()

        @jax.jit
        def step(p, s):
            loss, gradients = jax.value_and_grad(lambda x: jnp.mean(x['w'] ** 2))(p)
            updates, s = optimizer.update(gradients, s, p)
            return optax.apply_updates(p, updates), s, loss

        params, state, _ = step(params, state)
        expected = step(params, state)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = stream.write_checkpoint(root/'stream', params, state,
                self.metadata(), ModelFiles, serialization)
            core.write_checkpoint(root/'legacy', params, state,
                self.metadata(), ModelFiles, serialization)
            blob = (root/'stream/optimizer.msgpack').read_bytes()
            self.assertEqual(blob, (root/'legacy/optimizer.msgpack').read_bytes())
            self.assertEqual(manifest['files'], core.checkpoint_manifest(root/'legacy')['files'])
            restored = serialization.from_bytes(optimizer.init(params), blob)
            actual = step(params, restored)
            self.assertEqual(jax.tree.structure(expected), jax.tree.structure(actual))
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
                self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

    def test_optimizer_io_failure_keeps_incomplete_marker_and_refuses_resume(self):
        params, state, _ = self.state()
        def fail(output, *_):
            output.write(b'partial')
            raise OSError('Injected disk failure')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'checkpoint'
            with patch.object(stream, 'stream_optimizer', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'Injected disk failure'):
                    stream.write_checkpoint(root, params, state, self.metadata(), ModelFiles, serialization)
            self.assertTrue((root/'INCOMPLETE').exists())
            self.assertFalse((root/'READY').exists())
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                core.checkpoint_manifest(root)

    def test_existing_checkpoint_is_preserved_and_modified_bytes_are_refused(self):
        params, state, _ = self.state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'checkpoint'
            manifest = stream.write_checkpoint(root, params, state, self.metadata(), ModelFiles, serialization)
            with self.assertRaises(FileExistsError):
                stream.write_checkpoint(root, params, state, self.metadata(), ModelFiles, serialization)
            self.assertEqual(core.checkpoint_manifest(root), manifest)
            path = root/'optimizer.msgpack'
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                core.checkpoint_manifest(root)

    def test_short_optimizer_write_is_refused(self):
        class ShortWrite:
            def write(self, data): return max(0, len(data)-1)
        with self.assertRaisesRegex(OSError, 'Short optimizer checkpoint write'):
            stream.stream_optimizer(ShortWrite(), {'w': np.ones(4)}, serialization)


if __name__ == '__main__':
    unittest.main()
