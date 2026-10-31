## Full-shape GPU recovery verified; first epoch queued — 2026-10-31

| Work | Current state |
| :--- | :--- |
| Production-model baseline | ✅ Audited 8.46% WER over all2,694 retained validation clips |
| Full-shape CPU fallback | ✅ Three updates + READY snapshots; ~7.9s warm update |
| Full-shape GPU recovery | ✅ New process restored checkpoint1, completed updates2/3 and saved verified READY snapshots; terminal0 |
| First complete epoch | ⏳ Admission guard live, waiting above10% utilization limit; no child at this observation |
| Watch80 reading candidate | ✅ Target build and exact USB parity pass; unsigned/uninstalled, Watch79 boot/control remains separately owned |

**Poll session28705, guard1511822** (confirmed live in `/proc`).
Report: `build/training-readiness-20261031/gpu-full-epoch1-v2.json`.
Intent: `build/english-full-campaign-20261031/epoch1-v2-intent.json`.
Output, when admitted: `build/english-full-campaign-20261031/epoch1-v2`.
Frozen trainer7577539, verified GPU full-shape checkpoint3 +28,536 updates to
final28,539. Every training row remains selected. Checkpoint-every1000,
32GiB spare disk; about10.3GB of new checkpoint payload.
Full training keeps10% admission ceiling,30s dwell/3600s wait/43200s child cap,
host4GiB cap+1.5GiB reserve/1GiB floor, GPU2GiB pool/3GiB observed cap/2GiB reserve.
Other workloads continue. Never duplicate because observation expires.

The bounded GPU diagnostic session30596/guard1442926/child1447505 is
**terminal0**, both PIDs absent. Guard elapsed832.255s; training completed581.715s,
then250.540s process cleanup. Cold input verification took340.054s. Update2
(first execution after restore)70.110s; one warm update3 measured0.086s.
That is a diagnostic sample, not sustained throughput or epoch wall time.
Peak sampled host2,846,228,480B; ownedGPU2,296,381,440B.
[Audited full-shape GPU recovery](research/english-fullshape-gpu-recovery-20261031.json).
All model/Adam file hashes, source signatures, step metadata and all28,539-row
selection verified. CPU/CUDA signatures remain separate; no cross-backend
bit-identical gradient/weight claim. Original admission attemptv1 is terminal130,
never launched a child. No long CPU training job is running.

The baseline covers all retained validation clips; nine original long clips
were excluded during corpus preparation. No newly trained model has passed
recognition/native numeric/physical Watch gates or been deployed. The whole
goal remains **active and incomplete**. No Bak/audio/device/TUF operation here.

---

## English baseline complete; full-shape recovery probe running — 2026-10-31

| Work | Current state |
| :--- | :--- |
| Production-model baseline | ✅ Audited 8.46% WER: 4,538 edits / 53,635 words |
| Validation coverage | ✅ All 2,694 retained clips; nine longer original clips were excluded during preparation |
| CPU fallback | ✅ Three full-shape updates; terminal guard exit 0; all three checkpoint hashes verified |
| GPU full-shape recovery | 🔄 Bounded new-process checkpoint1 → updates2/3 probe running |
| First complete epoch | ⬜ Not started; original admission-only guard was terminated before launching a child |

GPU probe session **30596**, guard **1442926**, child **1447505** were confirmed
live in `/proc` at this handoff. Report:
`build/training-readiness-20261031/gpu-fullshape-resume3-v1.json`.
Frozen trainer source **7577539** and checkpoint1 are unchanged. This diagnostic
uses the existing 35% short-probe admission ceiling; full training retains 10%.
Two updates only, checkpoint every update, maximum 900 seconds execution.
Do not duplicate because observation expires. Inspect the exact guard and child.

CPU audit session69882 is terminal0: all three READY checkpoints, source hashes,
selected training rows and step/signature metadata verified.
[Audited CPU probe](research/english-cpu-fullshape-probe-20261031.json).
The training probe itself is terminal0, report
`build/training-readiness-20261031/cpu-fullshape-stream-v1.json`.
Steady CPU updates measured 7.927 and 7.901 seconds; roughly63 hours of update
calls alone per epoch. No long CPU epoch is running. CPU/CUDA execution
signatures differ; do not silently mix checkpoints across backends.

The older GPU campaign session19119 is terminal130/guard_interrupted, no child
was ever admitted. It was stopped to serialize our own heavy RAM jobs before
the CPU probe. The full goal remains **active and incomplete**, not paused.
Restore a uniquely named full-epoch admission attempt after the probe is terminal,
using the verified GPU checkpoint and unchanged 10% full-training threshold.
Other workloads stay running; no Bak work or research model deployment.

Watch77 remains the healthy fallback after another owner's Watch78 boot rollback.
That owner retains control and Watch79/seq80 boot-fix ownership. The reading UI
patch c66c86d is pushed separately with30Watch/27Deck/2notice fixtures passing;
no renderer firmware has been signed or installed. Preserve their release holds.

---

## First epoch queued; complete validation baseline running — 2026-10-31

| Work | Status |
| :--- | :--- |
| Watch 77 | ✅ OTA VALID; no device/control hold |
| GPU updates, recovery and full-shape checkpoint | ✅ Verified, frozen trainer7577539 |
| First complete epoch | ⏳ GPU admission waiting; no campaign child |
| Existing-model validation baseline | 🔄 238 /2694 clips at last observation; partial WER is not a final result |
| New-model quality and physical acceptance | ⬜ Still required; no research model deployed |

**Training:** session **19119**, guard **1251119**; report
`build/training-readiness-20261031/gpu-full-epoch1-v1.json`. Preserve 10% full-training
admission ceiling, other workloads, 30-second dwell and original3600swait.
Expected campaign continues verifiedfullshape checkpoint1 for28538updates,
checkpoint1000/sparedisk32GiB. Do not duplicate or lower a threshold to force admission.

**Baseline:** session **75240**, guard **1305361**, child **1305454**; all confirmed
live. Report `build/training-readiness-20261031/production-validation-full-v1.json`.
Output `build/heldout-native-20261031/production-validation-full-v1`.
Existing model8c956a95 only, complete validation split2694, no hints/truncation/
resampling. CPU1GiBcap/1GiBreserve/1GiBfloor/twoCPU/nice10/10800schildcap.
The actual nativeengine source matches3f1a1d8; model/engine frozen for the run.
No acoustic playback, microphone or device access.

New scorer `tools/eval_english_native.py` has9 complete locked-CPU controls and
actualfour-clip nativeintegration PASS. New test filename joins native discovery;
stdlib95discovered/77pass/18explicitdependency-skips, lockedCPU9scorer+5checkpoint
controls pass withnoskips. Skips do not establish audio/model acceptance.
Source66da322 contains the frozen scorer; later7b785d1 adds discovery compatibility.
[Evaluation method and controls](docs/english-evaluation.md).

Poll these exact handles/processes; never restart because observation expires.
Both original quality and fulltraining scope remain open. Corpus/test data have
not entered parameter fitting. Other workloads continue; no Bak/TUF/device changes.
The full Watch/English goal remains active and incomplete.

---

## Full-shape checkpoint passes; first epoch waiting — 2026-10-31

| Milestone | Status |
| :--- | :--- |
| Watch 77 | ✅ Installed OTA VALID; no device/control hold |
| GPU save/resume | ✅ Short-clip checkpoints 1/2 and new-process update/checkpoint 3 |
| Full shape | ✅ 25 seconds / 160 tokens; finite update + verified READY checkpoint |
| First complete epoch | ⏳ Admission state: waiting; no training campaign child at last observation |
| Learned native recognizer | ⬜ Epoch training, held-out/numeric quality and physical gates remain open |

Poll **session 19119**, guard **1251119**; child at observation **None**.
Report: `build/training-readiness-20261031/gpu-full-epoch1-v1.json`.
Intended output: `build/english-full-campaign-20261031/epoch1-v1`.
Intent: `build/english-full-campaign-20261031/epoch1-v1-intent.json`.
Frozen training source **7577539**. Continue full-shape checkpoint1 for
28,538 additional updates, final global28539, checkpoint-every1000 and32GiB
spare disk. Expected checkpoint payload about10.3GB thisepoch. Every training
row remains selected; no duration/token truncation or inherited base weights.

Full campaign uses **10%** admission ceiling, 30-second dwell, 3600-second
maximum admission wait, 43200-second (12h) owned execution cap. Memory: host4GiB
cap/1.5GiBlaunchreserve/1GiBruntimefloor, GPU2GiBpool/3GiBobservedcap/2GiBreserve.
Other workloads keep running. Do not duplicate on observation timeout; verify
actual guard/child and inspect last READY on any withdrawal. No export/deploy.

Full-shape probe79050/guard1240639/child1242319 is terminal0 (98.862s), READY1
hashverified, warmcache samplepeak2744172544B. Prior40536 did complete its
finite update, then hit the3.5GiBcap duringcheckpointwriting; NEVERpromote its
INCOMPLETEdirectory. [Full-shape proof](research/english-fullshape-checkpoint-20261031.json).
Full goal remains active and incomplete; no research model deployed.

---

## GPU checkpoints and recovery pass; full-shape probe live — 2026-10-31

| Milestone | Status |
| :--- | :--- |
| Watch 77 | ✅ Installed OTA VALID; no device/control hold |
| GPU short-clip updates | ✅ Fresh updates 1 and 2; READY model + Adam snapshots |
| New-process GPU resume | ✅ Restored 2 → completed 3; READY hashes verified |
| Planned full shape | 🔄 Probe state: running; all 28,539 clips retained |
| Useful trained recognizer | ⬜ Training, held-out/native quality and physical gates remain open |

Poll **session 40536**, guard **1207506**. Child at latest observation:
**1208963**. Verify the same process/report; do not duplicate on timeout.
Report: `build/training-readiness-20261031/gpu-fullshape-stream-v1.json`.
Output: `build/accelerated-student-20261031/gpu-fullshape-stream-v1`.
Frozen training source **7577539**. One fresh full-parameter update with
25-second / 160-token padding, limit-train-clips=0, full local audio cache.
Selection was independently rechecked: all 28,539 rows, zero duration/token
exclusions, exact planned ordered-ID hash. No truncation or inherited weights.

Same 3.5 GiB host cap / 1.5 GiB admission reserve / 1 GiB runtime floor,
GPU 2 GiB pool / 3 GiB observed cap / 2 GiB reserve, 35% short-probe ceiling,
30-second dwell, 900-second child cap. Full training retains its 10% ceiling.
Other workloads continue; no device, TUF or Bak changes.

Prior sessions 47919 and 4175 are terminal0; guards/children exited.
Fresh GPU checkpoints1/2 and restored3 are complete and hash verified. One
cross-run parameter differs by 2.3283e-10 while first loss/norms match exactly;
do not claim bitwise GPU reproducibility. The exact evidence is in
[the recovery receipt](research/english-gpu-checkpoint-recovery-20261031.json).
No research model has been deployed; full goal remains active and incomplete.

---

## Streamed GPU checkpoint retry admitted — 2026-10-31

| Milestone | Status |
| :--- | :--- |
| Watch 77 | ✅ Installed OTA VALID; device/control holds released |
| First CUDA student update | ✅ Finite gradients; nonzero update across all 29,537,627 trainable parameters |
| Original save attempt | ⛔ Host cap reached while serializing optimizer; partial checkpoint refused |
| Streaming optimizer writer | ✅ Five controls pass, exact Flax bytes and bit-identical next Adam update |
| Actual streamed GPU save/resume | 🔄 Admitted and running; save/resume not yet verified |

Poll **session 47919**, guard **1181715**, child **1189780**. Both are confirmed
live after admission. The full-geometry CPU comparison saved about 294 MiB
peak RSS and produced byte-identical 236 MB files; the actual GPU result is
still required.
Report: `build/training-readiness-20261031/gpu-student-stream-v1.json`.
Output if admitted: `build/accelerated-student-20261031/gpu-step2-stream-v1`.
Frozen source **7577539**, with `--stream-checkpoints` plus the existing optional
cache/heap-trim flags. The original CPU trainer is unchanged. The new flag and
writer hash are in the accelerated compatibility signature; do not resume an
older run under this new source. The old failed step has no valid READY snapshot.

Same 3.5 GiB owned host cap, now 1.5 GiB launch reserve and unchanged 1 GiB
runtime floor. GPU limits remain 2 GiB pool / 3 GiB observed cap / 2 GiB reserve;
35% applies only to this diagnostic, 30-second dwell and 900-second child cap.
Reuse the previous v3 compiler cache, since that actual update compiled, while
preserving all old partial artifacts. Other workloads continue uninterrupted.

Next: poll the same live handle/report. Verify two finite updates, terminal
guard and READY manifests before a new-process resume. Full 25s/160-token sizing,
meaningful training, native numeric/held-out speech and Watch physical gates
remain open. No research weights were deployed. Full goal remains active.

---

## Larger GPU diagnostic admitted; prior cap failure confirmed — 2026-10-31

| Work | Status |
| :--- | :--- |
| Watch 77 | ✅ Installed OTA VALID; all device/control holds released |
| Startup trim | ✅ About 280 MiB returned; all 337 model/optimizer arrays unchanged |
| Old 2.75 GiB test | ⛔ Terminal RSS limit at 3,129,143,296 bytes; zero updates/checkpoints |
| Larger diagnostic | 🔄 Live: 3.5 GiB host cap + 1 GiB admission reserve/runtime floor |
| Full training / native speech quality | ⬜ Not complete; no new weights deployed |

**Poll session 50246**, guard **1160207**, child **1161179**.
Both processes were verified live after admission. Report:
`build/training-readiness-20261031/gpu-student-heap-trim-v3.json`.
Output: `build/accelerated-student-20261031/gpu-step2-heap-trim-v3`.
Source `2210081`; training code remains frozen. Two full-parameter short-clip
updates and periodic READY checkpoints are requested. GPU budgets remain
2 GiB pool / 3 GiB observed cap / 2 GiB reserve, with the short-diagnostic
35% admission ceiling, 30-second dwell and 900-second child cap.
Other workloads continue. Full training retains its 10% utilization ceiling.

The launch reserve matches the earlier CPU control; this job has the stricter
3.5 GiB owned cap. Runtime withdrawal still preserves a 1 GiB host floor.
Do not duplicate on an observation timeout. Verify actual terminal state,
finite nonzero updates and READY files before new-process resume or full-shape
sizing. Session 30534 is terminal1, child -9, not live; its exact negative result
is in [the terminal receipt](research/english-heap-trim-terminal-20261031.json).
The full Watch/English goal remains active and incomplete.

---

## Watch 77 installed; full GPU student experiment live — 2026-10-31

✅ Private Watch 77/seq78 signed OTA is VALID. Its release preserves production
model, three apps, conversation profile, peer identity/security and journal.
Watch USB/control and target holds are released. No new microphone/speaker test
ran; no English research weights were deployed.

✅ New actual GPU initialization measurement: allocator trim returned
293,376,000 bytes (about 280 MiB), preserving all 337 model/optimizer leaves,
354,451,528 bytes, bit for bit. The digest also matches the earlier cache-only
control. Cache clearing alone did not reduce RSS. This proves immediate savings,
not the full training peak or speech quality.

⏳ Session **30534**, guard **1135513**, child **1137099** are confirmed live.
Source **2210081**, report
`build/training-readiness-20261031/gpu-student-heap-trim-v2.json`, output
`build/accelerated-student-20261031/gpu-step2-heap-trim-v2`.
First update is compiling; no update or READY checkpoint has been verified.
The stricter 2.75 GiB host cap matches the old failed compilation control;
1.5 GiB admission reserve and 1 GiB runtime floor remain. Same full model,
data, precision and math; no other workload interrupted. GPU limits remain
2 GiB pool / 3 GiB observed cap / 2 GiB reserve, 35% short-diagnostic admission,
900-second child cap. Full training remains unstarted.

Next: poll **30534** and these exact PIDs/report. Do not restart because an
observation times out. Check the terminal guard, update records and READY
checkpoint integrity before a new-process resume or full-shape probe.
[Actual initialization evidence](research/english-heap-trim-live-state-20261031.json).
Full Watch/English-model goal remains active and incomplete.

---

## Heap-trim GPU admission terminal; Watch 77 OTA underway — 2026-10-31

| Milestone | Status |
| :--- | :--- |
| Optional allocator-trim controls | ✅ 21 tests pass; next CPU Adam update is bit-identical |
| Full GPU memory experiment | ⏸️ Admission ended without a child; spare host RAM stayed below 5 GiB |
| Actual full-model RAM savings | ⬜ Unproven; the trim operation never ran on the GPU student |
| Full-corpus training | ⬜ Not started; no GPU student checkpoint created |
| Watch 77 | 🔄 Signed dev177 published and verified; one guarded OTA/install observation underway in private repo |

Session 39416 / guard 1107977 is terminal75 (`not-started`,
`host_memory_headroom`), confirmed absent from `/proc`. Do not poll it as live
or launch a duplicate automatically. See
[terminal admission receipt](research/english-heap-trim-admission-20261031.json).
All resource floors and other workloads were preserved. The optional trim
remains off by default. The full English-model goal remains active and incomplete.

Next: finish Watch 77 postflight, then wait for a sufficient Alienware compute
window. Real full-model updates/save/resume, full-shape resource sizing,
held-out speech quality and physical memory/latency remain required.

---

## Heap-trim experiment prepared; admission waiting — 2026-10-31

✅ Source `2210081` adds an opt-in allocator trim after initialization-cache
release. All 21 training audit/control/memory tests pass, including bit-identical
next Adam updates and refusal on an unsupported allocator. The original CPU
trainer and production firmware model remain unchanged. This is not yet a
proven full-model memory reduction.

⏳ Guard PID 1107977 / tool session 39416 is confirmed live, waiting for spare
host RAM. Report: `build/training-readiness-20261031/gpu-student-heap-trim-v1.json`.
No child at the latest observation. Same v5 budgets: 3.5 GiB host cap + 1.5 GiB
admission reserve; 1 GiB runtime floor; GPU 2 GiB pool / 3 GiB observed cap /
2 GiB reserve. The 35% utilization ceiling applies only to this short diagnostic.
Two full-parameter short-clip updates and checkpoint saving are requested;
full training remains unstarted. Other workloads keep running.

Next: poll the same handle/report and verify actual terminal state. Never
restart merely because an observation times out. If admitted, inspect all
state/RSS evidence, actual updates and READY checkpoint integrity before resume.
Watch 77 is built in the private repo; its release still awaits the other
owner's paired-device interval release. The full Watch/English goal is active.

---

## GPU admission finished; Watch integration continues — 2026-10-31

| Work | Status | What this proves |
| :--- | :--- | :--- |
| One-hour GPU window (v4) | ⏸️ Not started | Spare host RAM never met the 6 GiB launch threshold |
| Five-minute GPU window (v5) | ⏸️ Not started | Spare host RAM never met the 5 GiB launch threshold |
| Other workloads | ✅ Preserved | Neither window launched a training child or interrupted other work |
| Watch 77 integration | 🔨 Building in the private firmware repo | Current native engine, existing production model, preserved UI |
| New English model | ⬜ Not trained or deployed | GPU student updates, resume, full-shape sizing and held-out quality remain open |

Both admission reports are terminal `not-started` with `host_memory_headroom`.
The older entries below describing live guard 738565 are historical; no GPU
admission process remains from these two attempts. Do not restart either
finished handle. The v5 host cap was 3.5 GiB plus 1.5 GiB admission reserve;
the 1 GiB runtime floor stayed unchanged. No student checkpoint was created.

Next: finish the bounded Watch 77 compatibility build. Wait for free Alienware
resources before another explicitly scoped GPU job; keep other workloads running.
The complete English-model goal remains active and incomplete.

---

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
