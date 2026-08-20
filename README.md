# Telegraph WASM Scoring Module — Gaming-Resistant v2

Telegraph Protocol evaluation WASM that ranks Miner answer quality per intent.

**This is a drop-in replacement for the stock baseline with verified improvements.**

## Why this beats the baseline

The stock `telegraph-wasm-baseline` has three exploitable weaknesses that let a
low-quality miner inflate its score without improving answer quality:

1. **BM25 normalization bug (confirmed upstream).** The baseline divides the raw
   lexical score by `K1+1 = 2.5` per query term — the asymptotic bound as term
   frequency → ∞, which no real answer reaches. A perfect exact-match answer
   therefore caps at **0.40** instead of 1.0 (the baseline's own unit test
   `exact_match_scores_high` asserts `> 0.85` and *fails* against its own code).
   Our build fixes the normalizer so lexical overlap actually spans `[0, 1]`.
   That means ranking rewards are tied to real per-intent differences instead
   of a compressed, useless band.

2. **Length farming.** The stock length signal `sigmoid((len−50)/20)` rewards
   literally any filler: a 400-char padded answer beats a correct 60-char one.
   Replaced with `relative_length_quality(gt, answer)` — a log-ratio Gaussian
   peaked at answer length ≈ ground-truth length. Padded answers are dowmarked,
   tight correct answers score fully.

3. **Keyword stuffing.** BM25 lexical overlap never penalized repetition, so a
   miner could spam the ground-truth keywords. Added `stuffing_penalty` — a
   multiplicative factor combining distinct-token ratio and max-term share.
   Keyword-spammed garbage collapses toward a 0.2 floor.

All changes are **interface compatible**: same exports
(`rank_answer`, `rank_answer_cached`, `breakdown_answer`, `embed`,
`cosine_sim`, `bm25_score`, `alloc`, `dealloc`), same call signature
`(q_ptr, q_len, gt_ptr, gt_len, ma_ptr, ma_len)`.

## Build

```
cargo build --release --target wasm32-unknown-unknown --features real_weights
```

Output: `target/wasm32-unknown-unknown/release/telegraph_scoring.wasm` (24 MB).

> Note: `--features real_weights` is mandatory. The default feature set is
> `[]` which compiles the *projection* embedding mode (structurally correct but
> not semantically meaningful) — fine for pipeline tests, useless for judging.

## Test

```
cargo test --lib
```

Upstream cannot run `cargo test` (its `#[panic_handler]` collides with `std`'s
on the native target). We gated that handler behind `#[cfg(not(test))]` so the
native test build links. **26/26 tests pass**, including the upstream BM25
test that originally failed.

## Register (Track 2 submission)

1. Upload this `.wasm` via the integrate dashboard
   (`integrate.telegraphprotocol.com` → `/api/upload-wasm`), or host it where
   your API key can reference it.
2. Call `registerMiner` on-chain with the WASM hash/URL.
3. Update project details on the hackathon site (GitHub repo + description).

## What changed vs upstream baseline

- `src/antigame.rs` — new module: `stuffing_penalty`, `relative_length_quality`
- `src/bm25.rs` — fixed normalizer bug (upstream test now passes)
- `src/lib.rs` — composite applies the anti-gaming factors
- `src/allocator.rs` — panic handler gated out of native tests

Commit: `95318c5` `feat: add gaming-resistant evaluation signals`