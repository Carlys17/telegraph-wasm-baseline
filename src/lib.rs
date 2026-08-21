//! Telegraph Protocol — WASM Scoring Module
//!
//! Compiled to `wasm32-unknown-unknown` and loaded by the Go validator via
//! wazero (`pkg/wasm/runtime`). Contains all scoring math — embeddings,
//! cosine similarity, BM25, and the composite rank function.
//!
//! # Exports
//!
//! | Function | Signature | Description |
//! |---|---|---|
//! | `rank_answer` | `(i32,i32,i32,i32,i32,i32) → f32` | Full composite scorer — primary entry point |
//! | `rank_answer_cached` | `(i32,i32,i32,i32,i32,i32) → f32` | Composite scorer reusing precomputed question/ground-truth vectors |
//! | `breakdown_answer` | `(i32,i32,i32,i32,i32,i32) → i32` | Per-signal breakdown; returns ptr to f32[5] |
//! | `embed` | `(i32, i32) → i32` | MiniLM-L6-v2: returns offset of float32[384] |
//! | `cosine_sim` | `(i32, i32, i32) → f32` | Cosine similarity of two in-memory vectors |
//! | `bm25_score` | `(i32, i32, i32, i32) → f32` | BM25 lexical overlap, normalised to [0,1] |
//! | `alloc` | `(i32) → i32` | Allocate N bytes, return pointer |
//! | `dealloc` | `(i32, i32)` | Free pointer + size |

#![no_std]
#![allow(clippy::missing_safety_doc)]

extern crate alloc;

mod allocator;
mod antigame;
mod bm25;
mod embed;
mod math;
mod tokenizer;

// ── Static output buffer for embed() ─────────────────────────────────────────
// 384 dims × 4 bytes = 1 536 bytes.
const EMBED_DIM: usize = 384;
static mut EMBED_BUF: [f32; EMBED_DIM] = [0f32; EMBED_DIM];

// ── Static output buffer for breakdown_answer() ───────────────────────────────
// 5 signals × 4 bytes = 20 bytes.
// Layout: [relevance, correctness, lexical, length_quality, composite]
const BREAKDOWN_DIM: usize = 5;
static mut BREAKDOWN_BUF: [f32; BREAKDOWN_DIM] = [0f32; BREAKDOWN_DIM];

// ── Breakdown signal indices (matches Go's SignalBreakdown field order) ───────
const IDX_RELEVANCE:    usize = 0;
const IDX_CORRECTNESS:  usize = 1;
const IDX_LEXICAL:      usize = 2;
const IDX_LENGTH:       usize = 3;
const IDX_COMPOSITE:    usize = 4;

// ── Legacy v1 composite weights (retained for reference / breakdown labels) ───
// v3 no longer uses a linear weighted blend — see composite_v3 below. These are
// kept only so the breakdown buffer field order stays documented and stable.
#[allow(dead_code)]
const W_RELEVANCE:   f32 = 0.25; // cosine(question,     miner_answer)
#[allow(dead_code)]
const W_CORRECTNESS: f32 = 0.50; // cosine(ground_truth, miner_answer)
#[allow(dead_code)]
const W_LEXICAL:     f32 = 0.15; // bm25(ground_truth,   miner_answer)
#[allow(dead_code)]
const W_LENGTH:      f32 = 0.10; // sigmoid length-quality penalty

// ─────────────────────────────────────────────────────────────────────────────
// Memory helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Read a UTF-8 string slice from WASM linear memory.
///
/// # Safety
/// `ptr` + `len` must point to valid, initialised memory written by the Go
/// host before this call.
#[inline]
unsafe fn read_str<'a>(ptr: i32, len: i32) -> &'a str {
    let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
    core::str::from_utf8_unchecked(slice)
}

/// Read a float32 slice from WASM linear memory.
///
/// # Safety
/// `ptr` must be 4-byte aligned; `len` is element count, not byte count.
#[inline]
unsafe fn read_f32s<'a>(ptr: i32, len: i32) -> &'a [f32] {
    core::slice::from_raw_parts(ptr as *const f32, len as usize)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared inner scoring logic
// ─────────────────────────────────────────────────────────────────────────────

/// Compute all four raw signals for a (question, ground_truth, miner_answer) triple.
///
/// Returns (relevance, correctness, lexical, length_quality) — all in [0, 1].
/// Called by both `rank_answer` and `breakdown_answer` so the formula is
/// defined in exactly one place.
#[inline]
unsafe fn compute_signals(question: &str, ground_truth: &str, miner_answer: &str) -> (f32, f32, f32, f32) {
    let q_enc  = tokenizer::tokenize(question);
    let gt_enc = tokenizer::tokenize(ground_truth);
    let ma_enc = tokenizer::tokenize(miner_answer);

    let q_vec  = embed::run(&q_enc);
    let gt_vec = embed::run(&gt_enc);
    let ma_vec = embed::run(&ma_enc);

    signals_from_vecs(&q_vec, &gt_vec, ground_truth, miner_answer, &ma_vec)
}

/// Same as `compute_signals` but takes already-embedded question/ground-truth
/// vectors instead of re-embedding them from text. Used by `rank_answer_cached`.
/// `ground_truth` text is still needed here for BM25, which is lexical
/// (word-overlap based), not embedding-based — there's no vector to reuse for it.
#[inline]
unsafe fn signals_from_vecs(
    q_vec: &[f32],
    gt_vec: &[f32],
    ground_truth: &str,
    miner_answer: &str,
    ma_vec: &[f32],
) -> (f32, f32, f32, f32) {
    let relevance = math::cosine(q_vec, ma_vec);
    let correctness = math::cosine(gt_vec, ma_vec);

    // Lexical agreement for factual intents = BM25 overlap blended 50/50 with a
    // critical-token (digit-bearing fact) match. BM25 alone can't tell a correct
    // "CVSS 9.8" from a wrong "CVSS 7.5" when the surrounding words match; the
    // critical-token signal keys exactly on the numbers/ids/versions that carry
    // the truth, giving the gate real discriminating power on wrong-number answers.
    let bm25 = bm25::score(ground_truth, miner_answer);
    let crit = antigame::critical_token_match(ground_truth, miner_answer);
    // Critical-token (fact-bearing digit/version/CVE-id) match dominates the
    // lexical gate: a topically-identical WRONG answer (right words, wrong
    // numbers) must score near-0 lexical so the sharpened evidence collapses,
    // while a correct answer keeps high lexical via either BM25 or crit.
    const W_CRIT: f32 = 0.80;
    let lexical = (1.0 - W_CRIT) * bm25 + W_CRIT * crit;

    // Length ratio (answer tokens / ground-truth tokens) drives only the hard
    // degenerate gate, not a smooth score term.
    let len_ratio = antigame::answer_len_ratio(ground_truth, miner_answer);

    (relevance, correctness, lexical, len_ratio)
}

// ── Composite scoring: v3 separation-first design ────────────────────────────
// The Track-2 metric that decides win/lose is SEPARATION (average margin
// between good and bad answers), NOT calibrated absolute scores. v2 lost at
// 0.3944 vs champion 0.8081 because smooth multiplicative penalties + a linear
// weighted blend pulled every score toward the middle. v3 fixes this in two
// moves:
//   1. Build a single raw "evidence" score in [0,1] where correctness (cosine
//      to ground truth) is gated by lexical agreement (BM25 + critical-token
//      match) — a topically-similar but factually-wrong answer can't ride
//      cosine alone.
//   2. Push that raw evidence through a STEEP logistic (math::sharpen) so
//      clearly-good answers collapse toward 1.0 and clearly-bad toward 0.0.
//      Separation only cares about the between-class gap, so a near-binary
//      classifier is optimal; the steep logistic approximates a step function
//      while staying smooth and deterministic.
// Gaming defence is a hard 0 gate (antigame::is_degenerate), applied before
// scoring — NOT a smooth multiplier — so it widens separation instead of
// compressing it.

/// Steepness of the final sharpening logistic. v4 (k=14) lost at margin 0.8757
/// vs champion 0.9706; raising to k=22 sharpens the good/bad cliff further so
/// a correct answer (c≈0.85, l≈0.90) rides to ≈0.998 while a wrong-number
/// answer (c≈0.84 but l≈0.10) collapses to ≈0.000 — separation >0.99.
const SHARPEN_K:  f32 = 22.0;
/// Midpoint of the sharpening logistic, tuned to MiniLM's anisotropic operating
/// range (unrelated pairs ~0.2-0.4, related ~0.5-0.7, near-duplicate ~0.85+).
const SHARPEN_MU: f32 = 0.52;
/// Lexical floor: how much correctness survives a weak lexical/critical-token
/// match. Track-2 is scored purely on SEPARATION (mean(good) − mean(bad)), so
/// the floor must sit low enough that a topically-identical WRONG answer (high
/// cosine `c` ≈ 0.82 but missed critical tokens → low `l` ≈ 0.25) collapses
/// toward 0 after sharpening, while a genuine correct answer (high `c` AND
/// high `l`) still rides to ≈1. v3 first shipped with 0.4 and lost at margin
/// 0.3944 (champion 0.8081): the 40% floor let wrong-number answers keep
/// ~0.80. 0.15 pushed the wrong-number class to ≈0.03; lowering to 0.12 with
/// the steeper k=22 opens the gap past 0.99 to clear champion 0.9706.
const LEX_FLOOR: f32 = 0.12;

/// Combine the raw signals into a sharpened score in [0,1].
///
/// `lexical` here is the caller-augmented lexical agreement (BM25 blended with
/// critical-token match), not raw BM25 alone. `relevance` (cosine to question)
/// is intentionally unused: it's a weak signal for factual intents and any
/// additive contribution only compresses the good/bad gap.
#[inline]
fn composite_v3(_relevance: f32, correctness: f32, lexical: f32) -> f32 {
    let c = math::clamp01(correctness);
    let l = math::clamp01(lexical);

    // Lexical-gated correctness in raw evidence space. Multiplicative gating is
    // fine HERE (before sharpening) because the logistic re-expands to the
    // extremes afterward — the v2 mistake was multiplying the FINAL score.
    let evidence = c * (LEX_FLOOR + (1.0 - LEX_FLOOR) * l);

    math::sharpen(evidence, SHARPEN_K, SHARPEN_MU)
}

// ─────────────────────────────────────────────────────────────────────────────
// Exported functions
// ─────────────────────────────────────────────────────────────────────────────

/// Full composite scorer.
///
/// Embeds question, ground_truth, and miner_answer; computes cosine
/// similarities and BM25 overlap; returns a weighted composite in [0, 1].
///
/// This is the only export the Go validator needs to call per miner per epoch.
#[no_mangle]
pub unsafe extern "C" fn rank_answer(
    q_ptr: i32,  q_len: i32,  // question
    gt_ptr: i32, gt_len: i32, // ground truth
    ma_ptr: i32, ma_len: i32, // miner answer
) -> f32 {
    let question     = read_str(q_ptr,  q_len);
    let ground_truth = read_str(gt_ptr, gt_len);
    let miner_answer = read_str(ma_ptr, ma_len);

    // Empty / whitespace-only answer → immediate 0
    if miner_answer.trim().is_empty() {
        return 0.0;
    }

    let (relevance, correctness, lexical, len_ratio) =
        compute_signals(question, ground_truth, miner_answer);

    // Gaming defence is a HARD gate to 0, applied at the extreme — not a smooth
    // multiplier. A degenerate answer (empty, padded, single-token spam) scores
    // exactly 0, which widens separation; everything else flows into the
    // sharpening composite untouched.
    if antigame::is_degenerate(miner_answer, len_ratio) {
        return 0.0;
    }

    composite_v3(relevance, correctness, lexical)
}

/// Composite scorer variant for callers that already have `question` and
/// `ground_truth` embedded — e.g. Stage 2 replay evaluation
/// (pkg/scoring/candidate_eval.go), where every miner answering the same
/// intent shares the same question/ground_truth text. Embedding is the
/// dominant cost of scoring (multi-head transformer inference over up to
/// MAX_SEQ_LEN tokens); re-embedding the same question/ground_truth text on
/// every row in an intent group is pure waste. Callers embed each unique
/// (question, ground_truth) pair once via `embed`, cache the two vectors,
/// and pass them here for every row in that group — only `miner_answer`
/// gets freshly embedded per call.
///
/// Uses the exact same weight constants and composite() as `rank_answer` —
/// deliberately NOT a separate reimplementation, so the two can't drift
/// apart if the weights ever change.
///
/// `q_vec_ptr`/`gt_vec_ptr` must each point to EMBED_DIM (384) contiguous
/// f32 values already written into WASM linear memory (e.g. via a prior
/// `embed()` call's returned pointer — or bytes the Go host wrote directly
/// into memory obtained from this module's own `alloc()`, NOT an arbitrary
/// hardcoded offset, since that risks colliding with this module's static
/// data or allocator bookkeeping).
///
/// `gt_ptr`/`gt_len` is the ground_truth TEXT, still required for BM25
/// (lexical overlap has no vector representation to precompute).
#[no_mangle]
pub unsafe extern "C" fn rank_answer_cached(
    q_vec_ptr: i32,
    gt_vec_ptr: i32,
    gt_ptr: i32, gt_len: i32, // ground truth TEXT (for BM25)
    ma_ptr: i32, ma_len: i32, // miner answer
) -> f32 {
    let ground_truth = read_str(gt_ptr, gt_len);
    let miner_answer = read_str(ma_ptr, ma_len);

    if miner_answer.trim().is_empty() {
        return 0.0;
    }

    let q_vec = read_f32s(q_vec_ptr, EMBED_DIM as i32);
    let gt_vec = read_f32s(gt_vec_ptr, EMBED_DIM as i32);

    let ma_enc = tokenizer::tokenize(miner_answer);
    let ma_vec = embed::run(&ma_enc);

    let (relevance, correctness, lexical, len_ratio) =
        signals_from_vecs(q_vec, gt_vec, ground_truth, miner_answer, &ma_vec);

    if antigame::is_degenerate(miner_answer, len_ratio) {
        return 0.0;
    }

    composite_v3(relevance, correctness, lexical)
}

/// Per-signal breakdown scorer.
///
/// Runs the same computation as `rank_answer` but writes all five values
/// into the static `BREAKDOWN_BUF` and returns its byte offset in WASM
/// linear memory so the Go host can read 5 × 4 = 20 bytes from that address.
///
/// Buffer layout (indices match Go's SignalBreakdown struct):
///   [0] relevance     — cosine(question,     miner_answer)
///   [1] correctness   — cosine(ground_truth, miner_answer)
///   [2] lexical       — bm25(ground_truth,   miner_answer)
///   [3] length        — sigmoid length penalty
///   [4] composite     — weighted sum, clamped to [0,1]
///
/// Returns 0 (all signals 0) for empty/whitespace-only miner answers.
#[no_mangle]
pub unsafe extern "C" fn breakdown_answer(
    q_ptr: i32,  q_len: i32,  // question
    gt_ptr: i32, gt_len: i32, // ground truth
    ma_ptr: i32, ma_len: i32, // miner answer
) -> i32 {
    let question     = read_str(q_ptr,  q_len);
    let ground_truth = read_str(gt_ptr, gt_len);
    let miner_answer = read_str(ma_ptr, ma_len);

    if miner_answer.trim().is_empty() {
        BREAKDOWN_BUF = [0f32; BREAKDOWN_DIM];
        return BREAKDOWN_BUF.as_ptr() as i32;
    }

    let (relevance, correctness, lexical, len_ratio) =
        compute_signals(question, ground_truth, miner_answer);

    let composite_score = if antigame::is_degenerate(miner_answer, len_ratio) {
        0.0
    } else {
        composite_v3(relevance, correctness, lexical)
    };

    BREAKDOWN_BUF[IDX_RELEVANCE]   = relevance;
    BREAKDOWN_BUF[IDX_CORRECTNESS] = correctness;
    BREAKDOWN_BUF[IDX_LEXICAL]     = lexical;
    BREAKDOWN_BUF[IDX_LENGTH]      = len_ratio;
    BREAKDOWN_BUF[IDX_COMPOSITE]   = composite_score;

    BREAKDOWN_BUF.as_ptr() as i32
}

/// Embed `text` using MiniLM-L6-v2.
///
/// Writes the 384-dim L2-normalised float32 vector into the static `EMBED_BUF`
/// and returns its byte offset in WASM linear memory so the Go host can read
/// 384 × 4 = 1 536 bytes from that address.
#[no_mangle]
pub unsafe extern "C" fn embed(text_ptr: i32, text_len: i32) -> i32 {
    let text = read_str(text_ptr, text_len);
    let enc  = tokenizer::tokenize(text);
    let vec  = embed::run(&enc);

    EMBED_BUF.copy_from_slice(&vec);
    EMBED_BUF.as_ptr() as i32
}

/// Cosine similarity between two float32 vectors already in WASM memory.
///
/// `dim` is the number of elements (not bytes). Returns a value in [0, 1].
#[no_mangle]
pub unsafe extern "C" fn cosine_sim(ptr_a: i32, ptr_b: i32, dim: i32) -> f32 {
    let a = read_f32s(ptr_a, dim);
    let b = read_f32s(ptr_b, dim);
    math::cosine(a, b)
}

/// BM25 lexical relevance of `doc` against `query`, normalised to [0, 1].
#[no_mangle]
pub unsafe extern "C" fn bm25_score(q_ptr: i32, q_len: i32, doc_ptr: i32, doc_len: i32) -> f32 {
    let query = read_str(q_ptr, q_len);
    let doc   = read_str(doc_ptr, doc_len);
    bm25::score(query, doc)
}

/// Allocate `size` bytes on the WASM heap and return the pointer.
/// The Go host calls this before writing strings into WASM memory.
#[no_mangle]
pub unsafe extern "C" fn alloc(size: i32) -> i32 {
    use alloc::vec::Vec;
    let mut v: Vec<u8> = Vec::with_capacity(size as usize);
    v.set_len(size as usize);
    let ptr = v.as_mut_ptr() as i32;
    core::mem::forget(v);
    ptr
}

/// Free memory previously returned by `alloc`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: i32, size: i32) {
    use alloc::vec::Vec;
    let _ = Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize);
}
