## Isolated CUDA environment / GPU guard active — 2026-10-31

New GPU environment installation8818 TERMINAL0, all51 locked versions verified;
existing CPU core pins and absent CUDA plugin rechecked unchanged. No GPU backend
was initialized during installation. Paths/docs: docs/english-gpu.md and
research/english-gpu-admission-20261031.json.13 GPU guard tests (simulated GPU
telemetry, real owned/foreign process isolation) +4 context-binding tests pass.
Default admission:30s sampled idle <=10% utilization, no existing compute process,
GPU3GiB+2GiB reserve and host3GiB+2GiB launch reserve. Child JAXpool2GiB, usage
withdrawal thresholds sampled, not a hardware partition. Explicit compilation,
CUDA, Triton and XDG caches stay on workdata. Parent/device/receipt/pool/package
checks precede JAX import; backend smoke tests matmul, gradient and convolution.

First real window6471 TERMINAL75/not-started after180s, no child launched because
GPU0 stayed around24-25% utilization. Preserved receipt gpu-backend-guard-v1.json.
A NEW bounded one-hour wait is LIVE under exec61553, report
build/training-readiness-20261031/gpu-backend-guard-v2.json. Poll this exact handle
or verify its live guard PID; do not restart due to observation timeout. This is
not an admitted compute job yet. Child, if admitted, uses the new GPU environment
and writes gpu-backend-smoke-v2; observe its final guard AND numeric result before
claiming backend success. Other compute arrivals make this guard wait or stop
only its own group. No other workload is evicted.

Data stager exec2838/PID468768 remains LIVE, training archive still downloading;
validation+test5305 clips hash verified. Watch/Deck/model/checkpoints untouched.
Next finish admitted backend verification and add a GPU student path/checkpoint
acceptance; current student trainer remains CPU-only and training has NOT begun.
All original accuracy/numeric/physical gates remain open; goal remains active.

## Training readiness and full audio staging active — 2026-10-31

Goal remains active. New read-only auditor verifies every local file against the
full pinned corpus:4/33844 clips initially present (train2/validation2/test0),
all4 hashes correct. Missing33840 paths retained in build/training-readiness-20261031/audit-v1.
Four sparse/corrupt/escape audit tests and4 archive-integrity tests pass.

Official-PyPI CUDA13 dependency dry run6196 is TERMINAL0. Exact core CPU pins
preserved;51 wheel hashes committed in research/english-gpu-cuda13-linux-x86_64-py312.lock.txt.
No environment installation, GPU context, training or other-workload eviction.

Full corpus acquisition is LIVE under exec2838, stagerPID468768. Poll this exact
handle or live process; do not start a duplicate because a tool observation ends.
Status build/training-readiness-20261031/staging-v1/status.json gives the current
curl child PID and actual byte count. Official HTTPS archives validation->test->train,
8MiB/s ceiling, pinned bytes+MD5+SHA256 before regular-only extraction. Existing
probe cache preserved. New archive cache .system-cache/datasets/librispeech-20261031;
new audio .system-cache/ackerman-english-training/full-audio-20261031. Validation and test archives are now size/MD5/SHA-verified and extracted;
follow-up audit34066 TERMINAL0 verifies all2694 validation +2611 test clips
(10.63hours). Training archive download is live; full corpus is not ready yet.
All writes are on verified Alienware workdata; Bak untouched and its English hold respected.

Next implement GPU admission/separate locked environment while monitoring this
existing staging job; then repeat the full auditor on new audio after extraction.
The current trainer still forces CPU. Full training, strict numeric, held-out
speech, native memory/stack and physical Watch acceptance all remain open.
Read docs/english-training-readiness.md for the current human checklist.

## Composed-path C replay completed — 2026-10-31

Observed context perturbation5.071e-7 relative produces native projection change
0.00104577 at the eighth fixed decoder token. Reference vs perturbed-native
output differs3.453e-7 relative; error-vector cosine0.999999946. Native baseline
replay is bit-identical. The composed reference has one changed context code;
capturing intermediates is exact against original composition. Barrier after
normalization fails the same case and is NOT adopted as a fix.

Guards84805/57455 terminal0, peak composed process group1.40GB. New host tools
and exact receipts in research/english-cross-composition-20261031.json; human
scope and diagram in docs/english-composition.md. This supports cutoff sensitivity
for this case, not whole-model numeric/accuracy acceptance. Full fast gate stays
false. Move meaningful training preparation forward while retaining all controls;
no further subdivision of this one proven projection replay is required first.
No checkpoint, optimizer, policy, model/device or other-workload change.

## Recorded runtime restored / cross components controlled — 2026-10-31

A real evaluation mismatch was found: pinned WhistleConfig.from_saved drops
flash/dtype/remat/scan_unroll, and prior diagnostics used flash=True despite
training signature flash=False. New english_runtime restores and validates all
four options, checks geometry, and records the helper hash. Earlier default-mode
measurements remain historical evidence; use docs/english-runtime.md for scope.

Correct-runtime full fast comparison still fails all four clips. With identical
native normalized input, all cross-attention components and context Q8 codes
agree. Standalone normalization replay changes native query output only at
~3.16e-7 relative; it does not reproduce a large jump. Do not assert a specific
normalization/projection bug from these controls. Compare composed normalization
and quantization execution next, preserving all primitive controls and limits.

The source generator now adds13 capture calls, verified by5 integrity tests;
4 runtime-restoration tests and8 Zig ASAN/UBSAN probe checks pass. All previous
research artifacts, raw tensors and failed controls remain intact. The corrected
whole-float control now passes all4 clips, terminal0, under a3GiB cap/2GiB reserve
(observed1.41GB process group). Its initial unstarted
4GiB-cap guard was explicitly superseded, identity checked and confirmed terminal
before the replacement launched. No training child had started on that guard.
See research/english-recorded-runtime-20261031.json for exact receipts and final
float status. No model/firmware/checkpoint/policy changes or other-workload eviction.

## Cross-attention localized — 2026-10-31

The first-layer eighth-token discrepancy is now local to cross-attention:
shared-input relative error 0.000644707. Self-attention maximum 3.982e-7 and
feed-forward maximum 1.938e-8 pass. All 64 float sublayer comparisons pass.
Captured vs original training block output is exact in both modes; native
logits remain bit-identical. No acceptance limit changed. The next boundary
is cross-attention Q/K/V, cached clip memory and output projection.

Host-only generated source adds exactly five captures; four integrity tests
verify roundtrip-only-capture changes and stale/ambiguous anchor refusal.
Eight generated-probe ASAN/UBSAN checks pass. Numeric guards9446/95194 and
sanitizer build46255 are terminal0. No device, model, GPU or training-policy
change. Exact receipts: research/english-decoder-sublayers-20261031.json.

## Decoder layers localized — 2026-10-31

Active research narrows the eighth-token mismatch to decoder layer zero (the
first layer): shared-input relative error 0.000470491. The second layer agrees
on native input (maximum 3.063e-7); its cumulative drift is inherited. Expanded
native trace still reproduces retained logits bit for bit. Per-token traced vs
original simulation control passes stricter 2e-6 (measured max 1.485e-7).
All 32 float-layer comparisons and eight Zig ASAN/UBSAN probe checks pass.
Guards93453/63351/20746 and ASAN build79633 are terminal0. The first run's
whole-sequence control and source snapshot are retained alongside the second
run's stronger per-token check. No device/GPU/model/training-policy changes.
Next trace self-attention, cross-attention and feed-forward boundaries inside
first-layer eighth-token execution; native full-network gate still fails.
See docs/english-decoder.md and research/english-decoder-layers-20261031.json.

## Decoder boundary diagnosis completed — 2026-10-31

The actual C decoder reproduces retained logits bit for bit in fast and float
modes. On the Levi clip, the first seven forced-token hidden states agree;
the eighth fast hidden state diverges (relative 0.00028843). With identical
native hidden states, the output projection agrees (maximum 2.87e-6).
The float control passes all eight. Trace inside the eighth-token decoder next;
do not attribute its error to the output head or relax thresholds.

The new host probe passes actual-model execution and malformed-input/output
preservation checks under Zig ASAN/UBSAN. All decoder guards are terminal.
Attempt one failed because its probe binary had not built (cmake outside PATH);
its source/log/output are preserved. Pinned-CMake build, separate second output
and float control succeed. No model deployment or GPU/workload interference.
See `research/english-decoder-boundary-20261031.json` for exact evidence.

## Native-policy export controls completed — 2026-10-31

The active research continuation validated the actual native-policy step-two
export: all 89 packed matrices and all four float-network clips pass. All four
fast-network clips still fail the unchanged strict gate. Decoder comparison
using native encoder memory passes three clips but fails the Levi introduction
(0.003729 versus 0.002); investigate both encoder and decoder sensitivity.
See `research/english-native-policy-validation-20261031.json`.

Guarded jobs network-window-v1, float-window-v1 and weights-window-v1 are all
terminal. The fast comparison exits 1 with preserved failure evidence; float
and weights exit 0. No research model deployment, speech-accuracy claim,
GPU allocation or other-workload eviction. Earlier pauses below remain history.

# English native research resumed — parent goal active, 2026-10-30

## Current checkpoint — 2026-10-31

Research remains active. Parent resource choice: wait for free Alienware
resources; keep other workloads running. Bak remains held off.

Read [current feature/status guide](docs/english-native.md) and
`research/english-native-20261031.json`. The actual profile-3 mixed export is
8,832,915 bytes and fits the model partition. Two training steps and restart,
native/sanitized execution, six frontend comparisons and legacy regression pass.
Quantized activation/cache comparison, meaningful training/accuracy and physical Watch
acceptance remain open. No research firmware/model was installed.

Follow-on whole-network comparison found and fixed research position encoding.
A full-checkpoint audit then exposed NumPy/JAX quantization-boundary differences
in 12 of 89 CQ matrices. The corrected JAX-math export now passes all 89 weight
checks and the original encoder/decoder thresholds on all four clips. Preserve
`build/network-profile3-v{1,2}/` including tested binaries and reports. The
new artifact is `build/student-matched-profile3-v1/`, with successful receipts
in `build/weight-matched-v2.json` and `build/network-matched-v1/`. Read
`research/english-export-parity-20261031.json`. Next: compare the quantized
activation/cache path before meaningful training. No GPU or other workload was
displaced. Source trainer/features/precision hashes remain unchanged.

Fast-path comparison and diagnostics are now in
`docs/english-native-quantization.md` and
`research/english-native-quantization-20261031.json`. Original upstream A8/KV8
differs from this fork's actual grouped Hadamard/int8 CQ kernel. The proposed
native-aware simulation passes four actual matrix kernels and fixture gradient
checks, but strict whole-network comparison still fails. Shared-input controls
localize frontend rounding sensitivity: stem passes with identical native
features; seven of eight encoder blocks pass with identical native inputs.
Do not relax thresholds or call these native-policy-trained weights.

Preserve `build/network-fast-v1/`, `build/network-native-aware-v{1,2}/`,
`build/native-stages-v{1,2}/`, `build/native-quant-primitives-v{1,2}/` and
`build/native-quant-asan-v1/`. Older simulated module/checker source snapshots
and tested binaries stay with their original reports. The next task is to
bound remaining rounding sensitivity and prepare explicitly versioned native
training. The historical checkpoint and its exact trainer remain preserved at
Git commit `6fbd604`; the source below adds a new incompatible training signature.

The current trainer exposes `--deploy-policy native-fast` and hashes its policy,
module, activation domain and float-cache assumptions into checkpoints. Default
policy is still upstream A8/KV8. Thirty focused tests pass, including changed
policy/module refusal and real guard process-isolation/receipt preservation.
`tools/guard_english_job.py` provides bounded CPU admission without eviction.
The first native-policy step, cross-process model/optimizer restart and export
now pass. This two-clip probe is not meaningful training or speech acceptance.
Read `research/english-native-training-20261031.json` for final numeric receipts.

Queued 2026-10-31 under `build/native-policy-training-20261031/`:
`sensitivity-window-v1.json` runs the controlled block perturbation probe after
2.5 GiB is available; `training-window-v1.json` waits for 6 GiB before a fresh
one-step 4 GiB-cap native-policy training probe. Both retain a 1 GiB live host
floor and stop admission after five minutes. Inspect actual report/process
state before any retry; a waiting report alone is not proof of a live process.

All admission jobs above and `resume-window-v1.json`/`export-window-v1.json`
are now TERMINAL0 after memory freed naturally. Fresh outputs are
`step1-v1/step-00000001`, `step2-v1/step-00000002` and `export-step2-v1/` beneath
that same build root. Model SHA is
`37a8d6e407c7ffbc64fb46a8340cb41b29447a528fb836590e91800b94fc819b`.
Keep the effective model limit 11,010,048 bytes to preserve the signed app store.
The strict fast-path numerical gate remains open; do not mark it green from the
successful training machinery or controlled perturbation diagnosis.

Local preserved outputs beneath this worktree:
`build/english-native-training-20261031/stable-mixed-step{1-v4,2-resume}/`,
`build/student-mixed-profile3-v1/`, `build/frontend-mixed-profile3-v1/`,
`build/legacy-profile-comparison-v3/`, `build/admission-v4/`.
Resource launchers/logs remain in
`~/Projects/.agent-tmp/ackerman-english-native/`.

Trainer SHA `61e2a9d4b23cdfe7e26fcd1debf833287ffaf9c9164eb6e2e1b6d830f5898455`
must match when resuming step two. Feature and precision module hashes are in
the receipt. Preserve those source files, checkpoints and failed prior attempts.

The parent explicitly reactivated the full Watch goal. English-model native
compatibility research is resumed here. Preserve prior checkpoints and exact
training scripts, installed firmware and other owners' work. Bak shutdown and
service holds remain in force; do not start work there. The pause below is a
historical checkpoint for this scope. No research export is deployable without
held-out accuracy, format, memory and physical gates.

# Temporary pause — 2026-10-30

The parent requested a pause after the nearest completable milestone.
No new training, downloads, builds, deployment or device work until an explicit
resume. Preserve every checkpoint, failed attempt, source archive and cached clip.
This hold supersedes earlier resume notes in AGENTS.md.

The milestone is a bounded, full-parameter English training/checkpoint probe.
It is not a trained speech recognizer. See [design and remaining gates](docs/english-student.md)
and `research/english-neural-probe-20261030.json` for exact results.

## Retained on Alienware

| Item | Location beneath `/var/home/jack/Projects/` |
| :--- | :--- |
| Source | `Whistle-English-Training-20261030`, branch `feat/english-student-training-20261030` |
| Logs and checkpoints | `AckermanOS-whistle-decoder-memory/build/english-neural-20261030/` |
| First finite step | `.../step1-v4/step-00000001/` |
| Resumed step | `.../step2-resume/` |
| Exact launcher | `.agent-tmp/ackerman-watch/run_english_cpu_probe.py` |

Verified source and four-clip staging are in
`.system-cache/ackerman-english-training/`; the isolated Python environment is
`.system-cache/venvs/ackerman-english-training-20261030/`. The complete English
corpus/tokenizers remain on Bak under `/srv/ai/cache/datasets/librispeech/` and
`/srv/ai/artifacts/ackerman-os/english-corpus-20261029/`. Bak is held off by the
parent; do not power it on or retry jobs there during the pause.

## After explicit resume

1. Read this hold, host storage rules and canonical AckermanOS `release/COORDINATION.md`.
2. Verify hashes and READY markers before using the retained checkpoint. Choose new output paths.
3. Arrange resources for meaningful training; never evict occupied serving models without authorization.
4. Complete held-out speech, native/export and Watch acceptance before deployment.

Training optimizer/model binaries are intentionally local, outside Git. They are
preserved evidence, not an independent backup. No model or firmware was installed.
