#!/usr/bin/env python3
"""Generate a host-only instrumented copy; never edit the production engine."""
import argparse
import hashlib
import json
from pathlib import Path


def instrument(source):
    start = source.index('static void dec_block(')
    end = source.index('static void session_end(', start)
    block = source[start:end]
    anchors = {
        'input': '    /* Engram injection at its sites. */',
        'post-engram': '    /* Self attention with causal 3-tap conv on Q, K, V. */',
        'post-self': '    /* Gated cross attention over the clip (K, V precomputed per layer). */',
        'post-cross': '    mlp_residual(w, x, 1, L + D_PRE_HADA, L + D_MLP, 1.0f);',
    }
    for label, anchor in anchors.items():
        if block.count(anchor) != 1:
            raise ValueError('Changed decoder source anchor: '+label)
        call = f'    trace_boundary(layer, pos, "{label}", x, d);\n'
        block = block.replace(anchor, call+anchor)
    anchor = anchors['post-cross']
    block = block.replace(anchor, anchor+'\n    trace_boundary(layer, pos, "post-mlp", x, d);')
    def after(anchor, calls):
        nonlocal block
        if block.count(anchor) != 1:
            raise ValueError('Changed cross-attention source anchor: '+anchor)
        block = block.replace(anchor, anchor+'\n'+calls)
    after('    zc(w, x, d, X + X_NORM, xn);',
          '    trace_boundary(layer, pos, "cross-normalized-input", xn, d);')
    after('    sh_lin(w, &si, X + X_GATE, gt);',
          '    trace_boundary(layer, pos, "cross-raw-query", raw, nq);\n'
          '    trace_boundary(layer, pos, "cross-gate-logits", gt, nh * vd);')
    after('        zc(w, raw + h * qk, qk, X + X_QN, tmp);',
          '    trace_cross_head(layer, pos, h, tmp, qk);')
    anchor = '    lin(w, X + X_OUT, o, 1, t2);'
    if block.count(anchor) != 1:
        raise ValueError('Changed cross-attention output projection anchor')
    block = block.replace(anchor,
          '    trace_boundary(layer, pos, "cross-gated-context", o, nh * vd);\n'
          '    trace_projection_input(w, layer, pos, o);\n'+anchor)
    after(anchor, '    trace_boundary(layer, pos, "cross-projection", t2, d);')
    after('    zc(w, t2, d, X + X_POST, gt);',
          '    trace_boundary(layer, pos, "cross-post-norm", gt, d);')
    return source[:start]+block+source[end:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    source = a.source.read_text()
    generated = instrument(source)
    a.output.write_text(generated)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    a.receipt.write_text(json.dumps(dict(schema=1, scope=__doc__, production_source_sha256=sha(a.source.read_bytes()),
                                       generator_sha256=sha(Path(__file__).read_bytes()), generated_sha256=sha(generated.encode())), indent=2)+'\n')


if __name__ == '__main__':
    main()
