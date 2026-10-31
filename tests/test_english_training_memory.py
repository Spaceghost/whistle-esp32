"""Real JAX cache release preserves state and the next optimizer update."""
import os
from pathlib import Path
import sys
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
except ImportError:
    jax = None

import english_training_memory as memory


@unittest.skipIf(jax is None, 'Run in the locked training environment with JAX and Optax')
class InitializationCacheTests(unittest.TestCase):
    def test_next_real_adam_update_is_bit_identical_after_cache_release(self):
        self.check_next_update(trim=False)

    def test_next_real_adam_update_is_bit_identical_after_heap_trim(self):
        self.check_next_update(trim=True)

    def check_next_update(self, *, trim):
        params = {'weight': jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64}
        optimizer = optax.adamw(1e-4)
        state = optimizer.init(params)

        @jax.jit
        def step(p, s):
            loss, gradient = jax.value_and_grad(lambda v: jnp.mean((v['weight'] @ v['weight']) ** 2))(p)
            update, s = optimizer.update(gradient, s, p)
            return optax.apply_updates(p, update), s, loss

        p, s, _ = step(params, state)
        expected = step(p, s)
        receipt = memory.release_initialization_cache(jax, p, s, trim_host_heap=trim)
        actual = step(p, s)
        self.assertTrue(receipt['state_bit_identical'])
        self.assertGreater(receipt['state_bytes'], 0)
        if trim:
            self.assertTrue(receipt['host_heap_trim_requested'])
            self.assertTrue(receipt['glibc_version'])
            self.assertGreater(receipt['rss_after_state_verification_bytes'], 0)
        self.assertEqual(jax.tree.structure(expected), jax.tree.structure(actual))
        for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            self.assertEqual(np.asarray(x).tobytes(), np.asarray(y).tobytes())

    def test_digest_binds_shapes_and_dtypes_not_only_raw_bytes(self):
        a = np.arange(8, dtype=np.float32)
        self.assertNotEqual(memory.state_digest(jax, a), memory.state_digest(jax, a.reshape(2, 4)))
        self.assertNotEqual(memory.state_digest(jax, a), memory.state_digest(jax, a.view(np.int32)))

    def test_live_value_change_is_detected(self):
        self.assertNotEqual(memory.state_digest(jax, {'w': jnp.array([1., 2.])}),
                            memory.state_digest(jax, {'w': jnp.array([1., 3.])}))

    def test_unsupported_allocator_refused_before_touching_state(self):
        with patch.object(memory.ctypes, 'CDLL', return_value=object()):
            with self.assertRaisesRegex(ValueError, 'current glibc allocator'):
                memory.release_initialization_cache(None, None, None, trim_host_heap=True)


if __name__ == '__main__':
    unittest.main()
