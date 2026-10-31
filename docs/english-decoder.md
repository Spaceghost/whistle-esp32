# English decoder: where fast inference diverges

**The first decoder layer diverges at the eighth tested token.** The second
layer agrees when given identical input. The output head also agrees.

| Boundary | Result | What it establishes |
| :--- | :--- | :--- |
| Native probe vs original native logits | ✅ Bit-identical | The host trace preserves actual decoder behavior |
| First seven tested hidden states | ✅ Agree | The discrepancy appears later in this fixed sequence |
| First layer, eighth token, identical inputs | ☐ Relative error 0.00047049 | The mismatch is local to this layer |
| Second layer, identical inputs | ✅ Maximum relative error 3.07e-7 | Its cumulative difference is inherited |
| Output head, identical hidden states | ✅ Maximum logit difference 2.87e-6 | The earlier decoder error is not caused by the final projection |

```mermaid
flowchart LR
    A[Same native encoder memory] --> B[Decoder layer 1]
    B --> C[Eighth token diverges]
    C --> D[Layer 2 inherits drift]
    D --> E[Final logits miss strict gate]
    F[Same layer input] --> G[Layer 2 agrees]
    H[Same hidden state] --> I[Output head agrees]
```

These are **eight fixed comparison tokens**, not recognized words. The model
has only two optimizer steps. This diagnosis establishes neither useful speech
recognition nor Watch speed, memory or microphone performance. The full fast
network still fails; no research model has been deployed.

## How the comparison stays controlled

`host/decoder_probe.c` includes the existing engine and copies only the small
decoder orchestration to save lane states between unchanged production
operations. `tools/check_english_decoder.py --layers` refuses results unless
the final native logits reproduce the retained native run bit for bit.

`tools/english_decoder_layers.py` executes the pinned training model's layer
code. Each layer is compared in two ways: cumulative output and output using
the exact native layer input. This separates inherited drift from a local
discrepancy. The traced training calculation must also agree with the original
training calculation **at every token**, within a stricter 2e-6 relative control.
The measured maximum is 1.49e-7. Native acceptance remains 2e-4 relative / 0.002
absolute logits; those limits have not changed.

The earlier boundary-only probe is described in
[its exact receipt](../research/english-decoder-boundary-20261031.json).
It reproduced native execution in fast and float modes, localized the eighth
hidden-state error to 0.00028843, and passed all eight float hidden states.
The layer probe narrows that finding further.

The expanded probe passes actual-model execution and malformed-input/output
preservation checks under Zig ASAN/UBSAN. The float layer control passes all
32 comparisons. CPU guards were bounded to two cores, 3 GiB and five minutes;
all jobs completed without eviction or GPU allocation.

[Exact layer, token, sanitizer and resource receipts](../research/english-decoder-layers-20261031.json).

## Next action

Trace the eighth token **inside the first decoder layer**: self-attention,
cross-attention and feed-forward boundaries. Do not change acceptance limits
or attribute the discrepancy to the second layer/output head.
