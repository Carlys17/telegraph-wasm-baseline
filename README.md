# Telegraph Scoring Module — Separation-First (v3)

A WebAssembly evaluation script for the [Telegraph Protocol](https://telegraphprotocol.com) hackathon, **Track 2 (Script Author)**. It ranks Miner answers for a single canonical intent against ground truth, and is a drop-in replacement for the official Track 2 baseline.

> **Status — active scorer.** The `v3-separation-first` build is the live scorer for the `CVE_LOOKUP` intent on Telegraph (registration #152), promoted after beating the incumbent on Stage 2 separation.

## Contents

- [How Telegraph judges a scorer](#how-telegraph-judges-a-scorer)
- [Design](#design)
- [Results](#results)
- [Interface compatibility](#interface-compatibility)
- [Build](#build)
- [Test](#test)
- [Register](#register)
- [Project layout](#project-layout)
- [Changes vs the baseline](#changes-vs-the-baseline)
- [License](#license)

## How Telegraph judges a scorer

A candidate scorer replaces the current champion only if it beats it on **separation** — the average margin it places between good answers and bad ones. Scoring answers plausibly is not enough; the script must *rank* them more decisively than the incumbent while resisting attempts to game the score.

An earlier version of this scorer (v2, registration #147) lost on exactly this metric:

```
average margin 0.3944  vs  champion 0.8081
```

The sections below explain why that happened and how v3 corrects it.

## Design

### Why v2 lost

v2 pursued gaming-resistance with smooth, multiplicative penalties on the final score — a keyword-stuffing factor and a relative-length factor layered over a linear blend of cosine similarity, BM25, and length. Each of those factors pulls scores toward the middle of the range, so good and bad answers bunched together and separation collapsed. The anti-gaming logic worked against the metric being judged.

### What v3 does

v3 rebuilds scoring around one principle: **push good and bad answers to opposite extremes, and keep gaming defence off the smooth path.**

1. **Lexical-gated correctness.** Correctness — `cosine(ground_truth, answer)` — is gated by lexical agreement: BM25 overlap blended with a *critical-token match*, the fraction of the ground truth's fact-bearing tokens (CVSS scores, version strings, CVE IDs, dates — anything carrying a digit) that appear in the answer. MiniLM embeddings barely distinguish "CVSS 9.8 critical" from "CVSS 7.5 high", so a topically-correct but factually-wrong answer would otherwise ride cosine alone. The critical-token gate catches exactly that.

2. **Steep sharpening.** The gated evidence passes through a steep logistic (`k = 14`, `μ = 0.55`), turning the scorer into a near-binary classifier: clearly-good answers collapse toward `1.0`, clearly-bad toward `0.0`. Since separation only rewards the gap between classes, a sharp step function is optimal.

3. **Hard degenerate gate.** Empty, padded, and single-token-spam answers are forced to exactly `0` *before* scoring, rather than nudged down by a soft penalty. Applying gaming defence at the extreme widens separation instead of compressing it — the opposite of v2's mistake.

## Results

A native separation benchmark over representative answer classes (correct, wrong-number, off-topic, keyword-stuffed, padded):

```
v3   mean(good) = 0.981   mean(bad) = 0.089   margin ≈ 0.89
v2   mean(good) = 0.878   mean(bad) = 0.487   margin ≈ 0.39
champion reference margin = 0.8081
```

The benchmark uses representative operating-point values, not Telegraph's private Stage 2 test set, so it is directional evidence for the design, not a guarantee. The on-chain outcome — v3 active, v2 rejected — is the actual proof.

Reproduce:

```bash
rustc -O --edition 2021 tools/sep_bench.rs -o /tmp/sep && /tmp/sep
```

## Interface compatibility

Fully drop-in. The module exports the same symbols as the official baseline:

```
rank_answer   rank_answer_cached   breakdown_answer
embed         cosine_sim           bm25_score
alloc         dealloc
```

Same call signature `(q_ptr, q_len, gt_ptr, gt_len, ma_ptr, ma_len)`, same 384-dimensional MiniLM-L6-v2 INT8 embedding (magic `MLM2`), same linear-memory ABI. The binary is 24.2 MB, within the 32 MB limit.

## Build

```bash
cargo build --release --target wasm32-unknown-unknown --features real_weights
```

Output: `target/wasm32-unknown-unknown/release/telegraph_scoring.wasm`

`--features real_weights` is mandatory. The default feature set compiles a structurally-correct but semantically-meaningless projection embedding — fine for pipeline smoke tests, useless for judging.

## Test

```bash
cargo test
```

All 27 tests pass. The upstream `#[panic_handler]` collides with `std` on the native test target, so it is gated behind `#[cfg(not(test))]` to let the native test build link.

## Register

1. Open [integrate.telegraphprotocol.com](https://integrate.telegraphprotocol.com) → **Submit WASM**.
2. Use **Paste Link** with the release asset URL (a server-side fetch sidesteps the browser upload size limit), select the intent (`CVE_LOOKUP`), and confirm the on-chain transaction.
3. Stage 1 (structural checks) runs in seconds; Stage 2 (separation vs incumbent) takes a few minutes. Registry indexing lags roughly 2–3 minutes before the dashboard reflects status.

## Project layout

```
src/
  lib.rs         WASM exports + composite_v3 scoring
  antigame.rs    critical_token_match, is_degenerate (hard 0-gate)
  math.rs        cosine, sigmoid, sharpen (steep logistic)
  bm25.rs        BM25 lexical overlap, normalized to [0, 1]
  embed.rs       MiniLM-L6-v2 INT8 embedding
  tokenizer.rs   WordPiece tokenizer
  allocator.rs   no_std bump allocator
tools/
  sep_bench.rs   standalone separation benchmark
weights/
  minilm_l6_v2_q8.bin   quantized embedding weights (magic MLM2)
vocab.txt        tokenizer vocabulary
```

## Changes vs the baseline

| File | Change |
| --- | --- |
| `src/antigame.rs` | Added `critical_token_match` (fact-bearing token gate) and `is_degenerate` (hard 0-gate); removed the old smooth `stuffing_penalty` / `relative_length_quality`. |
| `src/lib.rs` | Added `composite_v3`: lexical-gated correctness + steep sharpening; gaming defence moved to a hard pre-score gate. |
| `src/math.rs` | Added `sharpen`, a steep logistic contrast transform. |
| `src/bm25.rs` | Fixed a normalization bug so exact-match lexical overlap spans `[0, 1]` instead of capping at 0.40. |
| `src/allocator.rs` | Panic handler gated out of the native test build. |
| `tools/sep_bench.rs` | Standalone separation benchmark. |

## License

[MIT](LICENSE). Fork of the official `telegraph-wasm-baseline`.
