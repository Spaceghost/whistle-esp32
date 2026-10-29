# Watch and English-model work temporarily paused

Parent requested a pause on 2026-10-29 after the nearest completable milestone.
New work requires an explicit parent resume instruction. Preserve all source,
installed state, cached audio, corpus archives, partial downloads and evidence.

| Checkpoint | State |
| :--- | :--- |
| ✅ Real speech fixtures | Four YouTube videos cached; local ASR 4/4 exact, silence empty |
| ✅ Decoder memory | Eight real-audio comparisons preserve transcript/tokens/confidence; 32,772 fewer heap bytes |
| ✅ English-corpus tooling | 15 admission/split guards; complete eight-suite CTest driver passes |
| ⏸️ Data and tokenizer research | Two evaluation archives verified; training archive partial preserved; no real tokenizer fit or neural training |
| ☐ Physical acceptance | Watch microphone, fallback route and post-decode stack remain open; music untouched |

The completed milestone is corpus/tokenizer **tooling and its guards**. It is not
an English-only speech model. Watch68 remains the last Watch installation
performed by this owner. No device or model changes occurred in this milestone.
Other device owners retain responsibility for their own in-flight operations.

## Preserved work

Natural-speech tools and evidence: private AckermanOS branch
`host/youtube-audio-cache-20261029`, GitHub `95b65dfe3c9402a6f485d06caf567671ee5e57cf`
(local `9befbede853eb9707e8228c9db4f650e685add01`) before this pause checkpoint.

English-corpus tooling: Whistle branch `feat/english-corpus-20261029`, worktree
`bak:/srv/ai/workspaces/ackerman-os/whistle-english-corpus-20261029`, based on the
installed decoder tree `763d56165396a9e295ce63a938d1defafe817ebf`.
Corpus cache: `bak:/srv/ai/cache/datasets/librispeech/`.
Receipts/logs: `bak:/srv/ai/artifacts/ackerman-os/english-corpus-20261029/`.
The `.part` archive is intentionally incomplete and is not accepted corpus data.

## Resume after the parent explicitly asks

1. Read the canonical repo's `PAUSED.md`, `AGENTS.md` and `release/COORDINATION.md`.
2. Resume the existing training archive with HTTP range continuation; verify the
   official MD5 `2a93770f6d5c6c964bc36631d331a522` and record SHA-256 before renaming it.
3. Run corpus preparation into fresh output paths; reject split overlap and preserve originals.
4. Resolve the missing Python.h development headers for the isolated Python
   3.14 SentencePiece build and smoke-test its
   isolated environment, then fit research vocabularies
   using training text only. Neural retraining and native/export compatibility remain required.
5. Recheck quiet and device ownership before one YouTube microphone test. Never
   pause music, reset a bond, overwrite a published firmware version or assume a host pass proves hardware acceptance.

[English corpus design and commands](docs/english-corpus.md).
