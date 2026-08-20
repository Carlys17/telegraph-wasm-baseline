# Telegraph WASM Scoring Module — Separation-First

Telegraph WASM scoring module: the program that judges how good a miner's
answer is. It embeds `question` / `ground_truth` / `miner_answer` with a
MiniLM-L6-v2 sentence transformer, gates semantic correctness with a lexical
fact-match signal, and pushes the result through a steep contrast transform so
that good and bad answers land at opposite ends of the scale.

This is a fork of the official `telegraph-wasm-baseline`. It keeps the same
exports, ABI, and embedding, and changes only the scoring maths. The baseline
combined cosine, BM25, and length into a linear blend; this module replaces
that blend with a separation-first composite (see [Scoring](#scoring)).

```
telegraph-wasm-baseline/
├── src/
│   ├── lib.rs         exports: rank_answer, rank_answer_cached, breakdown_answer,
│   │                  embed, cosine_sim, bm25_score, alloc, dealloc; composite_v3 scoring
│   ├── antigame.rs    critical-token fact match + hard degenerate gate
│   ├── embed.rs       MiniLM-L6-v2 inference (two modes, see below)
│   ├── tokenizer.rs   BERT-style tokenizer feeding embed.rs
│   ├── bm25.rs        single-document BM25 lexical scorer
│   ├── math.rs        cosine similarity, sigmoid, sharpen, L2 norm — pure libm
│   └── allocator.rs   no_std global allocator + panic handler
├── tools/sep_bench.rs   standalone separation benchmark
├── build.rs             compiles vocab.txt into a binary lookup table (real_weights mode)
├── vocab.txt            BERT uncased vocabulary (30,522 tokens)
├── weights/
│   └── minilm_l6_v2_q8.bin   INT8-quantized MiniLM-L6-v2 weights
├── Cargo.toml
└── Cargo.lock
```

Every source file carries a module-level doc comment explaining what it does
and why. Read the source directly for details.

## Scoring

The composite is built to *separate* answer quality, not just estimate it:

- **Lexical-gated correctness.** `cosine(ground_truth, answer)` is gated by a
  blend of BM25 overlap and a critical-token match — the fraction of the ground
  truth's fact-bearing tokens (numbers, versions, IDs, dates) present in the
  answer. This stops a topically-correct but factually-wrong answer from riding
  cosine similarity alone.
- **Steep sharpening.** The gated evidence passes through a steep logistic, so
  clearly-good answers collapse toward `1.0` and clearly-bad toward `0.0`.
- **Hard degenerate gate.** Empty, padded, and single-token-spam answers are
  forced to `0` before scoring, rather than reduced by a soft penalty.

`tools/sep_bench.rs` measures the good/bad margin over representative answer
classes:

```bash
rustc -O --edition 2021 tools/sep_bench.rs -o /tmp/sep && /tmp/sep
```

## Two build modes

**Projection mode (default)** — no real model weights, no Python. Token IDs are
hashed into a deterministic pseudo-embedding: same output shape as real
inference (384-dim, L2-normalised) but not semantically meaningful. Useful for
exercising the pipeline quickly, not for judging real answer quality.

```bash
rustup target add wasm32-unknown-unknown   # once
cargo build --release --target wasm32-unknown-unknown
```

**Real weights mode** — runs actual MiniLM-L6-v2 inference (6-layer
transformer, INT8-quantized) using `weights/minilm_l6_v2_q8.bin`. This is the
build to submit.

```bash
cargo build --release --target wasm32-unknown-unknown --features real_weights
```

Output: `target/wasm32-unknown-unknown/release/telegraph_scoring.wasm` (24.2 MB,
under the 32 MB limit).

## Test

```bash
cargo test
```

The upstream `#[panic_handler]` collides with `std` on the native test target,
so it is gated behind `#[cfg(not(test))]` to let the native test build link.

## Exports

```
rank_answer   rank_answer_cached   breakdown_answer
embed         cosine_sim           bm25_score
alloc         dealloc
```

Call signature `(q_ptr, q_len, gt_ptr, gt_len, ma_ptr, ma_len)`, 384-dimensional
MiniLM-L6-v2 INT8 embedding (magic `MLM2`), linear-memory ABI — identical to the
baseline.

## License

[MIT](LICENSE).
