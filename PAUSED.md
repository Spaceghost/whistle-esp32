## Current decoder and C API sanitizer revalidation complete — 2026-10-31

✅ Rebuilt profile/stage/generateddecoder with real GCC ASan/UBSan. Current
generated decoder8cases PASS: validactualmodel, NaN/misaligned/emptyencoder,
outofvocab/empty/misalignedtokens, repeatedoutputrefusal. Allstderr clean and
invalidcases createNOoutputs; repeatedvalid output inventory hash unchanged.
New check_english_decoder_safety.py requires currentpassing canaries and actual
ASan/UBSan symbols, records hashes and core dumpsdisabled.

✅ Seven native C API tests PASS under verifiedruntimes: EnglishAPI/commands,
fastallocationfallback,tokenizermemory,speechgatememory,decoderconfidence,
silentfrontend. ASan initpresent in ALL7binaries; sanitizerflags checked;6tests
assertions active (-UNDEBUG), commands use explicit failure returns. Five
trace-source integrity tests PASS, exactly13capture additions; production/
generator/generated hashes verified. Old historical Zig ASan labels are NOT
retroactively accepted; current decoder gap is closed by this NEW run. Other
untested profile/stage safety cases and physical/numeric/ASR gates stay open.

Guards35319 build,34840 decoder,6282 nativeC suite ALLTERMINAL0. No owned native
job remains. Exact research/english-native-sanitizer-revalidation-20261031.json;
human table/graph docs/english-decoder.md, updated docs/english-memory.md.
No engine/trainer/device/model/TUF/Bak/other-workload change. Full goal active.

GPUwait46639/guard738565 requires samehandle/livePID re-poll; last verified
waitingRAM,nochild. Do NOTduplicate on observation timeout. Requirements4GiBhost
budget+2GiBreserve, shortdiagnostic35percent graphics threshold,3600swait/900s
childcap. Full training10percent unchanged/NOTstarted. Next actual GPUstudent
updates/save/resume and full25s/160token sizing beforemeaningfultraining.

---

## Actual ESP32-S3 engine compile and ABI measured — 2026-10-31

✅ Baseline5d1fc97 and research engine4a547bc all5 component sources compile
with frozen Watch compile_commands/config, SDK5.5.4, Xtensa GCC14.2.0 and pinned
SDKimage209569617d40dc2190907bfa7508aa5dfc0823698663fc28a4405ef8012a78b0.
New check_english_target.py produces actualELF32Xtensa objects, sizes and .su
frames. Guard11379 TERMINAL0,62.14s; container removed, no active target job.
Container separate1GiBhard cap/no swap/twoCPU quota/180s timeout; outerguardRSS
only describes host client. Source/config mounted READONLY; outputs task scratch.

Unlinked text31410 ->33044B (+1634), initializeddata8B and BSS4168B unchanged.
Actual target nd_whistle state8456B BOTH (host8552 difference is ABI, NOT savings).
Largest frameenc_block20656B unchanged; decoderblock12992,decstep11568,mlprows
10336; logmel5232->5248. Allreportedframesstatic, but nested/callback/task stack
acceptance remains OPEN. This is NOT a fully linked firmware image, runtime
acceptance, new pin or model deployment. Exact research/english-target-compile-20261031.json;
human table/graph appended docs/english-memory.md. Engine/trainer unchanged.

Larger GPU admission46639 still needs re-polling samehandle/liveguard738565;
last confirmed waiting other compute/RAM, no child. Do not duplicate because
an observation expires. Requirements4GiBhost budget+2GiBreserve,900s childcap,
3600s wait. Short diagnostic35percent graphics tolerance; full training default
10percent and NOT started. All GPUstudent/save/resume/fullshape, native numeric,
ASR quality, full-link and physical WatchRAM/stack/speed/mic gates open. Goal active.

---

## Native requested heap measured / sanitizer evidence corrected — 2026-10-31

✅ New host wh_memory and checker run actual8,832,915B native-policy export on
1/4/8s synthetic silence through open/encode/decode/close. Requested heap peaks
1,824,500 /2,139,004 /4,169,028B; final tracked live0, no allocation failures.
8s plus Watch two128000-float capture buffers =5,193,028B arithmetic, NOT physical
PSRAM/headroom acceptance. Static hostmodelstate8552B, stack/allocator/UI/network/
apps and physical heap placement excluded. Silence emits1token; not ASR quality
or long-decoder coverage. Exact research/english-native-memory-20261031.json;
human table/graph docs/english-memory.md. Original engine/trainer untouched.

⚠️ Found earlier Zig ASAN claims unsupported: combinedflags heap-overflowcanary
exits0; current old profile/stage/decoder binaries have UBSan symbols, no ASan.
Single-ASan Zig canary failslink; signed-overflow UBSan canary aborts correctly.
Earlier docs corrected; historical receipts retained, their ASAN scope needs
revalidation (new probe does not retroactively cover generated/malformed cases).
✅ Real GCC16.2.1-2.fc44 ASan/UBSan runtimes extracted to workdata cache, noOS
install. OfficialKoji identity/digests/SHA checked; no RPM signature claim. New
check_sanitizer_runtime.py detects BOTH intentionalfaults, core dumps disabled.
Actual model1/4/8s GCC ASan+UBSan run terminal0, all metrics EXACT release match,
no sanitizer errors. Guards93263 terminal127 missing cmakePATH retained; corrected
97988 build0,47372 run0,79664 ZigUB build0,79690 ZigUB run0,27181 GCCsanbuild0,
10794 GCCsanrun0. All owned native profiling jobs terminal.

Larger GPU admission wait46639 is still LIVE, guardPID738565 reverified, report
build/training-readiness-20261031/gpu-student-window-v4.json, no child admitted.
Needs4GiBhost budget+2GiBreserve; wait up to3600s, childmax900s. Otherlimits and
short-diagnostic35percent graphics tolerance unchanged, cacheclearOFF. Poll same
handle/livePID, never duplicate on observation timeout. Full training still
default10percent and NOT started.

Watch source inspection: vendor engine5d1fc976b0b630406b1511161707c74f37348030
predatesAWE1. Future target integration/pin+model compatibility/migration needed
AFTER trained quality evidence; no firmware/model/app/device/TUF changes here.
GPU student/save/resume/full-shape, full native numeric, held-out speech and
physical Watch RAM/stack/speed/mic gates remain OPEN; goal active.

---

## Cache experiment terminal; larger diagnostic admission wait LIVE — 2026-10-31

GPUv3 exec72921 TERMINAL1, child-9 on owned RSS cap after87.30s, no completed
update/checkpoint. Clearing caches preserves ALL337 GPU model+Adam leaves /
354,451,528 bytes, but does NOT fix2.75GiB cap. Immediate RSS2084913152 ->
2085613568B; later sampled peak3,085,082,624B. It stays opt-in/off by default,
not an adopted memory fix. Three CPU tests pass incl exact next Adam update.
Final receipt research/english-initialization-cache-control-20261031.json.
Earlier live student/CPU-test notes below are superseded. Source6486e8c pushed.

A NEW bounded one-hour admission wait is LIVE under exec46639, report
build/training-readiness-20261031/gpu-student-window-v4.json. Current guardPID
is recorded in that receipt and was verified live at handoff. No child admitted.
Diagnostic requests4GiB host RSS budget +2GiB launch reserve,1GiB host floor;
retains2GiB JAX pool/3GiB observedGPU cap/2GiB VRAM reserve,2CPUsnice10,35percent
short-diagnostic graphics ceiling,900s max child runtime. Cache release OFF.
Full meaningful training retains10percent utilization default; NOT started.
Poll46639 or exact live PID; do not restart because an observation expires.
Child if admitted writes build/accelerated-student-20261031/gpu-step2-v4.
Require final guard AND training/checkpoint evidence before claiming success.

Next: short GPU updates/save/resume, then FULL25s/160token shape resource test
covering all28,539training clips before meaningful training. All native numeric,
held-out ASR, Watch RAM/stack/physical gates OPEN; backend CUDA proof remains
valid. No other workload eviction, Bak action, device/model/app/TUF/conversation
change. Full Watch/English goal active; no research deployment.

---

## GPU student compiler memory measured — 2026-10-31

Student exec65627 TERMINAL1, child exit-9 via owned-group RSS guard. First
full-model update compilation reached3,040,595,968B RSS against2.75GiB cap;
VRAM2,267,021,312B,345.25s. NO completed update or READY checkpoint. Earlier
RUNNING notes below superseded; no active GPU job remains. Exact receipt
research/english-gpu-student-memory-20261031.json. Successful backend proof
remains valid; no other workload/device/model/conversation changed.

Next experiment --release-initialization-cache records current RSS and requires
bit-identical hashes of every live model/optimizer array across JAX cache clear.
New runner/helper source and option are signed into checkpoint compatibility;
existing CPU checkpoints require their recorded source. Original CPU trainer
and native quantization unchanged. CPU regression94482 TERMINAL0: all3 tests pass, including bit-identical
next Adam update after clearing caches. GPU memory benefit and student execution
still unverified. Full goal remains active, no research deployment.

---

## Actual CUDA backend verified — 2026-10-31

✅ Sixth diagnostic guard53317 terminal0, backend all_passed. RTX3070: matmul
maxabs5.722e-6, gradient3.576e-7, convolution2.384e-7, existing limits unchanged.
No CPU fallback. All51 installed packages and immutable admission/checker/lock
hashes revalidated. 42.75s, peak1.316GB hostRSS /2.250GB VRAM. Diagnostic host
cap tightened to2GiB, keeping2GiB launch reserve/1GiB runtime floor;35percent
utilization ceiling was diagnostic-only. Earlier false backend flags below
are historical. See research/english-gpu-backend-20261031.json.

✅ Full training plan retains every28,539 clip/100.59h with25s/160token padding,
identical ordered selection to30s/320tokens. No exclusions/truncation; longest
24.525s/150tokens. One batch1 epoch28,539 updates visits each clip once.
Exact plan research/english-full-training-plan-20261031.json.

A bounded two-update GPU student diagnostic is RUNNING under exec65627, report
build/training-readiness-20261031/gpu-student-window-v2.json. Prior student
window19266 was superseded before a child, terminal130. Diagnostic v2 uses
35percent graphics tolerance and a tighter2.75GiB host cap with2GiB reserve;
full training retains default10percent. Uses2 short public clips/2 updates and
periodic checkpoints with verified backend proof. This is machinery scope,
not meaningful full-corpus training. Poll actual handle or verify live guardPID;
observation timeout is not terminal and must not trigger a duplicate.
GPU student/resume, full training, native numeric/accuracy and physical Watch
gates remain OPEN; goal active, no research model deployment or device change.

---

## Current milestone — full corpus and durable accelerated CPU runner — 2026-10-31

✅ All 33,844 required audio hashes pass: training 28,539 / 100.59h;
validation 2,694 / 5.31h; test 2,611 / 5.32h. All three source archives passed
size, MD5 and SHA-256 checks and extraction. Stager2838 terminal0; audit-v3
reports complete, no missing/corrupt files. Earlier LIVE staging notes below
are historical and superseded.

✅ Separate accelerated runner saves periodic model + Adam snapshots. Fresh
CPU updates1–2 produce READY checkpoints; all29,537,627 parameter bytes and
optimizer bytes exactly match the unchanged original CPU reference. A new
process restores step2 and completes update/checkpoint3. Twelve control tests
pass. Initial missing-import failure retained; corrected source31d294d.
Guards22057/73071/19563 terminal0; original CPU trainer/native engine unchanged.
See docs/english-accelerated-training.md and exact research receipt.

⏳ GPU student execution remains unverified. Guard61553 was superseded before
any child and ended130. Diagnostic44522 (35% graphics tolerance only, default
memory budgets unchanged) ended75/not-started: host RAM below5GiB launch need.
No backend child/CUDA initialization. All earlier LIVE GPU notes below are
superseded. Full training still uses10% utilization policy. Separate CUDA env
installed, all51 versions match; other workloads keep running.

Next validate real CUDA backend when RAM is free, then GPU student periodic
save/resume. Meaningful full-corpus training, strict native numeric gate,
held-out speech, Watch memory/stack and physical acceptance remain OPEN.
Goal active, no research model deployment. Watch/Deck/apps/conversations unchanged.
Evidence: research/english-full-corpus-readiness-20261031.json and
research/english-accelerated-checkpoints-20261031.json.

---

Historical milestone log follows; current state above takes precedence.

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
