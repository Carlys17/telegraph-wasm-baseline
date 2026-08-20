# Telegraph WASM Scoring Module — Separation-First (v3)

An evaluation script (WASM) for the [Telegraph Protocol](https://telegraphprotocol.com) hackathon,
**Track 2 (Script Author)**. It scores and ranks Miner answers for a single canonical intent against
ground truth.

This repo is a fork of the official Track 2 baseline (`telegraph-wasm-baseline`), extended to rank
answer quality more sharply while staying resistant to gaming.

**Status:** the `v3-separation-first` build is the **active scorer** for the `CVE_LOOKUP` intent on
Telegraph (registration #152), having beaten the incumbent on Stage 2 separation.

---

## How Telegraph judges an evaluation script

A candidate scorer is hot-swapped in only if it beats the current champion on **separation** — the
average margin it puts between good answers and bad ones. It is not enough to score answers
plausibly; the script has to *rank* them more decisively than the incumbent, and resist attempts to
game the score.

An earlier version of this scorer (v2, registration #147) **lost** on exactly this metric:

```
your average margin 0.3944 vs champion 0.8081
```

The rest of this document is why that happened and how v3 fixes it.

---

## Why v2 lost, and what v3 does differently

### Root cause of the v2 failure

v2 tried to be gaming-resistant with **smooth, multiplicative penalties** on the final score
(a keyword-stuffing factor, a relative-length factor) layered on top of a linear weighted blend of
cosine + BM25 + length. Every one of those factors pulls scores toward the middle of the range. The
result: good and bad answers bunched together, and separation collapsed. The anti-gaming logic was
actively working against the thing being judged.

### v3 design (separation-first)

v3 rebuilds scoring around a single principle: **push good and bad answers to opposite extremes, and
keep gaming defence out of the smooth path.**

1. **Lexical-gated correctness.** Correctness (`cosine(ground_truth, answer)`) is gated by lexical
   agreement — BM25 overlap blended with a **critical-token match**: the fraction of the ground
   truth's fact-bearing tokens (CVSS scores, version strings, CVE IDs, dates — anything carrying a
   digit) that appear in the answer. MiniLM embeddings can't distinguish "CVSS 9.8 critical" from
   "CVSS 7.5 high" — they embed almost identically — so a topically-correct but factually-wrong
   answer would otherwise ride cosine alone. The critical-token gate catches exactly that.

2. **Steep sharpening.** The gated evidence is pushed through a steep logistic
   (`k = 14`, `μ = 0.55`), turning the scorer into a near-binary classifier: clearly-good answers
   collapse toward 1.0, clearly-bad toward 0.0. Separation only rewards the gap between classes, so a
   sharp, smooth step function is optimal.

3. **Hard degenerate gate (not a soft penalty).** Empty, padded, and single-token-spam answers are
   forced to exactly **0** *before* scoring. Applying gaming defence at the extreme **widens**
   separation instead of compressing it — the opposite of v2's mistake.

### Measured effect

A native separation benchmark over representative answer classes (correct, wrong-number,
off-topic, keyword-stuffed, padded):

```
v3: mean(good)=0.981  mean(bad)=0.089  margin ≈ 0.89
v2: mean(good)=0.878  mean(bad)=0.487  margin ≈ 0.39
champion reference margin = 0.8081
```

The benchmark uses representative operating-point values, not Telegraph's private Stage 2 test set,
so it is directional evidence of the design — not a guarantee. The on-chain result (v3 active, v2
rejected) is the actual proof.

Reproduce: `rustc -O --edition 2021 tools/sep_bench.rs -o /tmp/sep && /tmp/sep`

---

## Interface compatibility

Fully drop-in. Same exports as the official baseline:

```
rank_answer, rank_answer_cached, breakdown_answer,
embed, cosine_sim, bm25_score, alloc, dealloc
```

Same call signature `(q_ptr, q_len, gt_ptr, gt_len, ma_ptr, ma_len)`, same 384-dim MiniLM-L6-v2
INT8 embedding (magic `MLM2`), same linear-memory ABI. Binary is 24.2 MB, under the 32 MB limit.

---

## Build

```
cargo build --release --target wasm32-unknown-unknown --features real_weights
```

Output: `target/wasm32-unknown-unknown/release/telegraph_scoring.wasm`

`--features real_weights` is mandatory. The default feature set compiles a structurally-correct but
semantically-meaningless projection embedding — fine for pipeline smoke tests, useless for judging.

## Test

```
cargo test
```

27/27 pass. The upstream `#[panic_handler]` collides with `std` on the native test target, so it is
gated behind `#[cfg(not(test))]` to let the native test build link.

---

## Register (Track 2 submission)

1. Open `integrate.telegraphprotocol.com` → **Submit WASM**.
2. Use **Paste Link** with the release asset URL (server-side fetch avoids the browser upload size
   limit), select the intent (`CVE_LOOKUP`), and confirm the on-chain transaction.
3. Stage 1 (structural checks) runs in seconds; Stage 2 (separation vs incumbent) takes a few
   minutes. Registry indexing lags ~2–3 min before the dashboard reflects status.

---

## What changed vs the baseline

- `src/antigame.rs` — `critical_token_match` (fact-bearing token gate) and `is_degenerate`
  (hard 0-gate); the old smooth `stuffing_penalty` / `relative_length_quality` are gone.
- `src/lib.rs` — `composite_v3`: lexical-gated correctness + steep sharpening; gaming defence moved
  to a hard pre-score gate.
- `src/math.rs` — added `sharpen` (steep logistic contrast transform).
- `src/bm25.rs` — fixed a normalization bug so exact-match lexical overlap spans `[0, 1]` instead of
  capping at 0.40.
- `src/allocator.rs` — panic handler gated out of the native test build.
- `tools/sep_bench.rs` — standalone separation benchmark.
