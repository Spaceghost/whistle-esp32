# Smaller English model: data and vocabulary work

**Research resumed 2026-10-30 · No replacement speech model produced or deployed.**

The parent reactivated the Watch/English goal. [The previous pause checkpoint](../PAUSED.md)
is retained as history; unrelated holds remain in force.

| Stage | State | Completion evidence |
| :--- | :--- | :--- |
| ✅ Corpus preparation code | Checks official archive checksums, paired text/audio and disjoint splits | Nine focused corpus tests |
| ✅ Tokenizer admission code | Rejects changed manifests, non-English metadata and split leakage; actual native trainer smoke passes | Six admission tests plus native integration; all nine CTest suites pass |
| ✅ English speech corpus | 28,539 train / 2,694 validation / 2,611 test clips admitted | Source hashes and speaker/utterance/encoded-audio overlap checks pass |
| ✅ Smaller English vocabularies | 8,192 / 4,096 / 2,048-piece candidates fitted on training text only | All scored text views round-trip exactly with zero unknown tokens |
| ☐ Real English speech model | Neural retraining, export, native compatibility and accuracy acceptance remain | No candidate may replace the locked firmware model yet |

```mermaid
flowchart TD
  S[Official English speech archives] --> C[Check published checksum and record SHA-256]
  C --> P[Pair original FLAC and transcript]
  P --> L[Reject speaker, utterance or encoded-audio overlap]
  L --> T[Training split only: fit smaller BPE vocabularies]
  L --> H[Validation and test splits: score after fitting]
  T --> H
  H --> N[Future neural training and export]
  N --> E[Native engine compatibility and held-out speech accuracy]
  E --> W[Watch memory, stack and microphone acceptance]
```

## Corpus source

[LibriSpeech, OpenSLR12](https://www.openslr.org/12/) supplies read English audio
with aligned transcripts under CC BY 4.0. Credit Vassil Panayotov, Guoguo Chen,
Daniel Povey and Sanjeev Khudanpur, *LibriSpeech: an ASR corpus based on public
domain audio books* (2015). Original license text is retained during extraction.
Use `train-clean-100` for training, `dev-clean` for validation and `test-clean`
for held-out evaluation. Published archive MD5 values are pinned from the
[official checksum file](https://openslr.trmal.net/resources/12/md5sum.txt).
Local SHA-256 receipts identify exact cached bytes. This is source-data integrity,
not a firmware trust/signing mechanism.

```sh
python3 tools/prepare_english_corpus.py \
  --archive-dir /srv/ai/cache/datasets/librispeech \
  --audio-dir /srv/ai/cache/datasets/librispeech/english-corpus-20261030 \
  --output /srv/ai/artifacts/ackerman-os/english-corpus-20261029/prepared-20261030
```

Use fresh output paths; interrupted attempts are retained with `INCOMPLETE` and
must not be trained on. Original files and transcripts are never truncated.
Clips over the configured duration limit are excluded whole. Default limit is
30 seconds, matching Whistle's training window; an eight-second eligible count
is also recorded for the Watch capture limit. Speaker/utterance/hash leakage
checks include excluded clips. FLAC sample rate, channels, bit depth and sample
count are checked through STREAMINFO. Preparing manifests does not fully decode
every audio file or independently re-audit every spoken word.

JSONL rows include `audio`, `text`, `language: en`, source identity, speaker and
audio SHA-256. The manifests retain original casing. Their receipt records
counts, retained hours, excluded clips and manifest hashes. These files contain
public corpus data, never family conversations or notebook uploads.

## Vocabulary experiment

The working isolated environment is Python 3.12.16 with SentencePiece 0.2.2,
under `/srv/ai/cache/venvs/ackerman-english-py312-20261030/`. System Python and
global DNS were not changed. Package/interpreter hashes, the compiler overrides
and the offline build patch are in [the resumed evidence](../evidence/english-research-resumed-20261030.json).
The patch makes CMake use Abseil already bundled in the verified source archive;
no library C++ source was changed. `CC`, `CXX` and `LDSHARED` explicitly select
Bak's installed GCC rather than the interpreter's original Clang build defaults.

`tools/check_english_tokenizer.py` exercises the real trainer with the same
options as the corpus job. It catches native assertions that pure metadata
tests cannot detect. Byte fallback passes unseen-name, UTF-8 and apostrophe
round trips with zero unknown tokens. Do not add unused `required_chars`: the
pinned BPE implementation asserts when those characters have zero frequency.
The CTest driver includes this integration check when SentencePiece is available.

```sh
/srv/ai/cache/venvs/ackerman-english-py312-20261030/bin/python tools/train_english_tokenizer.py \
  --corpus /srv/ai/artifacts/ackerman-os/english-corpus-20261029/prepared-20261030 \
  --output /srv/ai/artifacts/ackerman-os/english-corpus-20261029/tokenizers-20261030
```

This trains new BPE tokenizers using training-split text with deterministic
original/lowercase/sentence-case views. Validation and test text never fit the
vocabulary. Special IDs remain pad0/eos1/bos2/unk3; byte fallback preserves
unseen names and UTF-8 text. All held-out text views must round-trip exactly
without unknown tokens. [SentencePiece options](https://github.com/google/sentencepiece/blob/master/doc/options.md)
explain these controls. Exact package version, options, input and output hashes
are retained; reproducibility is scoped to those pinned tools and platform.

New token IDs change Whistle's embedding and Engram-history semantics. These
tokenizers cannot be substituted into existing weights. Current native language
prefix geometry and signed model guards also require explicit adaptation for a
new vocabulary. The report therefore marks firmware compatibility and deployment
permission false. Existing English-conditioned model bytes remain unchanged.

Reported row/logit savings are arithmetic estimates at the current row format,
not an exported whole-model measurement. A smaller vocabulary can require more
tokens per word and increase decoding work. Those costs must be measured against
accuracy, latency, flash and RAM after neural training. English source text alone
does not prove that multilingual neural knowledge has been removed.

Physical speech tests continue to use the separately cached YouTube clips.
Read-audiobook results do not establish spontaneous child-speech accuracy.

## Measured vocabulary trade-off

| Text vocabulary | Validation tokens/word | Test tokens/word | Maximum test tokens | Same-format embedding estimate saved |
| :--- | ---: | ---: | ---: | ---: |
| 8,192 | 1.309 | 1.314 | 126 | 0 bytes |
| 4,096 | 1.483 | 1.495 | 143 | 1,081,344 bytes |
| 2,048 | 1.732 | 1.749 | 172 | 1,622,016 bytes |

[Complete corpus and tokenizer evidence](../evidence/english-vocabularies-20261030.json).
Training retains 100.59 hours from 251 speakers. Validation and test each use
40 different speakers; nine over-30-second clips per evaluation split were
excluded whole. Source transcripts and audio were preserved.

These are new English-text vocabularies, not pretrained speech models. Their
IDs differ even where their size equals the original tokenizer. Fewer rows
reduce an embedding estimate, while more tokens can increase decoder work.
Neural training, actual export size, recognition quality, latency and target
RAM/stack measurements must decide between them. No vocabulary is selected
for deployment from these text statistics alone. Output hashes also include
training paths/tool metadata; a new run records new immutable receipts.

A separate [untuned two-bit embedding experiment](embedding-quantization.md)
saved 1,048,704 model bytes while worsening the Levi introduction. That candidate
is rejected for deployment. Proper quantization-aware training remains open.
