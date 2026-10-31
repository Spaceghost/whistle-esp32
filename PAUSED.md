# English native research resumed — parent goal active, 2026-10-30

## Current checkpoint — 2026-10-31

Research remains active. Parent resource choice: wait for free Alienware
resources; keep other workloads running. Bak remains held off.

Read [current feature/status guide](docs/english-native.md) and
`research/english-native-20261031.json`. The actual profile-3 mixed export is
8,832,915 bytes and fits the model partition. Two training steps and restart,
native/sanitized execution, six frontend comparisons and legacy regression pass.
Whole-network numerical parity, meaningful training/accuracy and physical Watch
acceptance remain open. No research firmware/model was installed.

Follow-on whole-network comparison found and fixed research position encoding:
the encoder now passes the original numerical threshold on all four clips.
Decoder logits still exceed 0.002; do not relax the threshold. Preserve
`build/network-profile3-v{1,2}/` including tested binaries and reports. The
next bounded task is to isolate that decoder discrepancy. Neither comparison
used a GPU or disturbed other workloads.

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
