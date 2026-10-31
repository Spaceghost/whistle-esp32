"""Release initialization executables while verifying live training state."""
import gc
import ctypes
import hashlib
import json
from pathlib import Path


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise ValueError('Current RSS unavailable')


def state_digest(jax, state):
    import numpy as np
    leaves, structure = jax.tree.flatten(state)
    digest = hashlib.sha256(str(structure).encode())
    total = 0
    for leaf in leaves:
        # Transfer one leaf at a time, rather than making a full host copy.
        array = np.asarray(leaf)
        if not array.flags.c_contiguous or array.dtype.hasobject:
            raise ValueError('Expected contiguous numeric training arrays')
        header = json.dumps([array.dtype.str, list(array.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(memoryview(array).cast('B'))
        total += array.nbytes
        del array
    return digest.hexdigest(), total, len(leaves)


def release_initialization_cache(jax, params, optimizer, *, trim_host_heap=False):
    trim = None
    glibc_version = None
    if trim_host_heap:
        # Resolve only the current process allocator; never load a second libc.
        libc = ctypes.CDLL(None)
        try:
            version = libc.gnu_get_libc_version
            trim = libc.malloc_trim
        except AttributeError as exc:
            raise ValueError('Host heap trimming requires the current glibc allocator') from exc
        version.argtypes, version.restype = [], ctypes.c_char_p
        glibc_version = version().decode('ascii')
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    state = (params, optimizer)
    jax.block_until_ready(state)
    before_digest = state_digest(jax, state)
    before_rss = rss_bytes()
    jax.clear_caches()
    gc.collect()
    cache_rss = rss_bytes()
    trimmed = bool(trim(0)) if trim is not None else False
    after_rss = rss_bytes()
    after_digest = state_digest(jax, state)
    if before_digest != after_digest:
        raise ValueError('Training state changed while releasing initialization caches')
    record = dict(state_sha256=before_digest[0], state_bytes=before_digest[1],
                state_leaves=before_digest[2], state_bit_identical=True,
                rss_before_bytes=before_rss, rss_after_bytes=after_rss,
                rss_reduction_bytes=before_rss-after_rss)
    if trim_host_heap:
        record.update(host_heap_trim_requested=True, glibc_version=glibc_version,
                      allocator_reported_release=trimmed,
                      rss_after_cache_release_bytes=cache_rss,
                      rss_after_state_verification_bytes=rss_bytes())
    return record
