"""Matched CQ policy: four-bit tied embedding, two-bit body (or all four-bit)."""
TIED_EMBEDDING_BITS = 4


def qat_params(params, quantize, body_bits):
    if body_bits not in (2, 4):
        raise ValueError('Supported English QAT body widths are 2 and 4')
    result = quantize.cq_ste_params(params, body_bits)
    if body_bits != TIED_EMBEDDING_BITS:
        result['embedding']['embedding'] = quantize.cq_ste(params['embedding']['embedding'], TIED_EMBEDDING_BITS)
    return result


def pack_matrix(export, matrix, bits, group):
    """Same pinned CQ arithmetic, with its existing two-bit codebook enabled."""
    import numpy as np
    if bits not in (2, 4) or group != 128:
        raise ValueError('Unsupported English packing policy')
    w = np.asarray(matrix, np.float32)
    cb = export._cq_codebook_np(bits, group)
    H = export._cq_hadamard_np(group)
    out, width = w.shape
    pad = (-width) % group
    padded = np.pad(w, ((0, 0), (0, pad))) if pad else w
    groups = padded.reshape(out, padded.shape[1] // group, group).astype(np.float32)
    rotated = groups @ H
    norm = np.sqrt((rotated ** 2).sum(-1, keepdims=True))
    unit = rotated / np.maximum(norm, 1e-12)
    indices = export._nearest_idx(unit, cb).reshape(out, padded.shape[1])
    return export._pack_lsb(indices, bits), norm[:, :, 0].astype(np.float16)


def pack_model(export, params, config, tokenizer, audio, body_bits):
    if body_bits not in (2, 4):
        raise ValueError('Unsupported English QAT body width')
    original = export._q
    def q(name, matrix, bits, group):
        import numpy as np
        selected = TIED_EMBEDDING_BITS if name == 'embedding' else bits
        matrix = np.asarray(matrix, np.float32)
        packed, norms = pack_matrix(export, matrix, selected, group)
        return export._Tensor(name, export.CQ, matrix.shape,
            packed.tobytes() + norms.tobytes(), group, selected)
    export._q = q
    try:
        return export._pack_cact(params, config, body_bits, 128, tokenizer, audio=audio)
    finally:
        export._q = original
