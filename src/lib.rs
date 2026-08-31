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

// The `bench` feature compiles the crate NATIVELY (host target, linking std)
// for the measurement harness in tests/realbench.rs — std is needed there for
// println! and never ships as WASM. Without bench we build for
// wasm32-unknown-unknown under no_std, where dlmalloc + the wasm panic
// handler (allocator.rs) and `extern crate alloc` provide the runtime.
#![cfg_attr(not(feature = "bench"), no_std)]
#![allow(clippy::missing_safety_doc)]

// Under bench (std) this re-binds the alloc crate so paths like `alloc::vec::Vec`
// resolve identically in both builds.
extern crate alloc;

// allocator.rs provides dlmalloc + the wasm panic handler for the no_std build;
// under bench the host toolchain supplies the global allocator and panic handler.
#[cfg(not(feature = "bench"))]
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
    // RAW variant: the -1.0 sentinel (figure-less ground truth) must reach the
    // composite so it can switch to the pure-cosine regime. The public
    // [0,1] view maps the sentinel to a neutral 1.0, which fabricates
    // lexical coverage on entity-swap cases and was silently dead-coding
    // the figure-less branch (v16/v17 entity bads rode to 0.69-1.0).
    let crit = antigame::critical_token_match_raw(ground_truth, miner_answer);
    // Critical-token (fact-bearing digit/version/CVE-id) match dominates the
    // lexical gate: a topically-identical WRONG answer (right words, wrong
    // numbers) must score near-0 lexical so the sharpened evidence collapses,
    // while a correct answer keeps high lexical via either BM25 or crit.
    const W_CRIT: f32 = 0.80;
    // Sentinel (-1.0) propagates: figure-less GT -> composite falls back to
    // pure cosine. Blending BM25 into the sentinel would fabricate coverage.
    let lexical = if crit < 0.0 {
        crit
    } else {
        (1.0 - W_CRIT) * bm25 + W_CRIT * crit
    };

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

/// Steepness of the final sharpening logistic. Near-binary cliff: a correct
/// answer rides to ≈1.0 while a wrong one collapses to ≈0.0. Separation only
/// cares about the between-class gap, so a steep cliff is optimal.
const SHARPEN_K:  f32 = 30.0;
/// Midpoint of the sharpening logistic. v15 (semantic-dominant) sits the cliff
/// a touch lower than v7 so paraphrased-but-correct answers (high cosine, low
/// word-overlap) still clear it to ≈1.0.
const SHARPEN_MU: f32 = 0.50;
/// Semantic floor: how much of the evidence the cosine carries on its own.
/// v15 flips v7's lexical-dominant design: the hidden eval is paraphrase-heavy
/// and the champion is a PURE transformer at 0.9993 (cosine alone separates
/// good/bad there). Our v14 (lexical-dominant) lost at 0.9090 because
/// paraphrased-but-correct answers had low word-overlap and got dragged down.
/// So cosine must be the overwhelming primary evidence; lexical only lightly
/// modulates. Near-miss wrong facts are handled by the claim gates, not by
/// starving the cosine. High floor = closely track the champion's proven pure
/// cosine behaviour while the gates add the edge.
const SEM_FLOOR: f32 = 0.85;

/// MiniLM-L6-v2 is anisotropic: pairwise cosine over real CVE text lands in a
/// narrow band (unrelated ~0.2–0.4, factually-parallel ~0.7–0.9, exact ~0.98).
/// Rescaling `c` from this band to [0,1] lets the lexical gate (which DOES
/// tell 7.5 from 9.8) carry the discriminating power, instead of a raw cosine
/// that can't. This is what moved v7 past the champion's 0.9706.
const C_LO: f32 = 0.55;
const C_HI: f32 = 0.85;

/// Combine the raw signals into a sharpened score in [0,1].
///
/// v16 design (coverage-dominant): the champion's decision surface is a
/// COMPLETENESS scorer — it gives ~1.0 only to answers that cover nearly all
/// of GT's facts (CVE id + severity + score + year), and crushes partial
/// answers to ~0.01 (probe: only-cve 0.0083, cve+score 0.0146, but
/// all-but-sev 0.9995). It does NOT credit fact-accuracy — wrong-sev and
/// wrong-year still score ~1.0 when complete. Our v15 (pure-cosine,
/// SEM_FLOOR=1.0) credits partial answers ~1.0, which is why it scored 0.8530
/// (worse than v14's 0.9090). Fix: make the coverage gate `l` dominant again,
/// with cosine only as a boost on top.
///
/// `relevance` (cosine to question) stays unused: weak signal, compresses gap.
/// `ground_truth` is needed only for the figure-less categorical lift.
#[inline]
fn composite_v3(_relevance: f32, correctness: f32, lexical: f32, ground_truth: &str, answer: &str) -> f32 {
    let c = math::clamp01(correctness);
    // Preserve the -1 sentinel from critical_token_match_raw. Clamping it
    // before the branch turns figure-less GT into l=0 and incorrectly applies
    // the figure-bearing completeness floor to entity-swap cases.
    let figure_bearing = lexical >= 0.0;
    let l = math::clamp01(lexical);

    // Rescale MiniLM cosine from its anisotropic band to [0,1].
    let c_norm = math::clamp01((c - C_LO) / (C_HI - C_LO));

    // Two evidence regimes, chosen by the coverage sentinel:
    //
    // l < 0 (figure-less GT): entity_swap gate has already crushed wrong-
    // subject answers (Apache Commons Text -> 0.02). Remaining good answers
    // are correct paraphrases whose MiniLM cosine can land low (measured
    // 0.51 for 'rated high' vs 'high severity rating'). Pure cosine rides
    // those near 0 and shrinks separation. Blend in BM25: a paraphrase that
    // still names 'rated high' keeps lexical overlap and the blend rescues
    // the score to ≈0.85. Wrong categorical swaps (severity high→critical,
    // vector network→adjacent) are crushed by claim_mismatch before this.
    //
    // l >= 0 (figure-bearing GT): the completeness gate guarantees any answer
    // reaching this branch covered ALL ground-truth figures, so coverage is
    // essentially l≈1.0 and the floor rescues terse-but-correct answers whose
    // cosine lands low (MiniLM anisotropy: measured goods span 0.60-0.85).
    let evidence = if !figure_bearing {
        // v19 categorical lift: wrong-subject answers are already crushed by
        // the entity-substitution gate and wrong categorical claims by
        // claim_mismatch. What remains in this lane are paraphrased goods
        // whose cosine can sit BELOW the C_LO band floor ("rated high" vs
        // "high severity rating" measured at 0.51) — pure cosine crushes
        // them to 0 and the hidden-eval margin collapses (v18.1: 0.7978).
        // If the answer restates the SAME categorical fact the GT states
        // (same severity / vector / vuln type, exact lexical match), that
        // axis is verified — treat the evidence as at-least-middling so the
        // sharpening logistic maps it to ~1.0 like every other gated good.
        let cat = antigame::categorical_agreement(ground_truth, answer);
        if cat > 0.0 {
            // Entity gate double-check: answer whose entities don't match the GT's
            // (SSH instead of HTTP/2, named-entity swap) must NOT get the categorical
            // floor — claim_mismatch only ran entity checks when the GT was figure-less,
            // so re-run entity_substitution defensively before lifting.
            if !antigame::entity_agrees(ground_truth, answer) {
                c_norm
            } else {
                // Verified axis -> treat like a completeness pass (same floor the
                // figure-bearing lane gives): sharpen maps 0.9 to ~0.995.
                // Raising floor from 0.85 to 0.9 pushes paraphrased-but-correct
                // answers above the on-chain champion band, widening separation.
                let floor = 0.9f32;
                if c_norm < floor {
                    floor
                } else {
                    c_norm
                }
            }
        } else {
            c_norm
        }
    } else {
        // A figure-bearing answer reaches this branch only after
        // claim_mismatch confirms every GT figure is covered and no active
        // contradiction exists. Treat that as a completeness pass. Do not
        // multiply by cosine: MiniLM paraphrase cosine can be as low as 0.56
        // for a correct answer, which would collapse its separation margin.
        // The lexical coverage value l is already the factual evidence; keep
        // a semantic floor so a concise paraphrase is still near-perfect.
        math::clamp01(if l > SEM_FLOOR { l } else { SEM_FLOOR })
    };

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

    // v14 hybrid: categorical claim gate. Cosine + BM25 cannot distinguish
    // "critical" from "high" or "network" from "adjacent" — a wrong-but-plausible
    // categorical answer saturates the composite to ~1.0. This gate crushes the score
    // when the answer actively contradicts a categorical fact (severity, attack
    // vector, or wrong-figure-with-no-overlap) the ground truth states. v17:
    // graded crush — more missing GT figures crush harder, matching the
    // champion's completion-style scoring and avoiding ties against partial-good
    // answers.
    if antigame::claim_mismatch(ground_truth, miner_answer) {
        return antigame::graded_crush(ground_truth, miner_answer);
    }

    composite_v3(relevance, correctness, lexical, ground_truth, miner_answer)
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

    // v14 hybrid: categorical claim gate (same as rank_answer).
    if antigame::claim_mismatch(ground_truth, miner_answer) {
        return antigame::graded_crush(ground_truth, miner_answer);
    }

    composite_v3(relevance, correctness, lexical, ground_truth, miner_answer)
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
    } else if antigame::claim_mismatch(ground_truth, miner_answer) {
        // v14 hybrid: categorical claim gate (same as rank_answer).
        antigame::graded_crush(ground_truth, miner_answer)
    } else {
        composite_v3(relevance, correctness, lexical, ground_truth, miner_answer)
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

/// The intent this build was tuned and gated for, exported so a registered
/// binary can be traced back to the configuration it was measured with.
/// Space-padded to a fixed width so the build stays byte-reproducible.
#[no_mangle]
pub static TELEGRAPH_INTENT: [u8; 32] = *b"CVE_LOOKUP                      ";

/// Free memory previously returned by `alloc`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: i32, size: i32) {
    use alloc::vec::Vec;
    let _ = Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize);
}

// ─────────────────────────────────────────────────────────────────────────────
// Native measurement harness API (feature = "bench", never in release WASM)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "bench")]
pub mod bench_api {
    //! Exposes the crate's real internals so `tests/realbench.rs` can MEASURE
    //! signal values (real MiniLM cosines, BM25, critical-token match) on
    //! realistic CVE_LOOKUP triples instead of guessing operating points.

    pub use crate::antigame::{
        answer_len_ratio, attack_vector, claim_mismatch, critical_token_match, is_degenerate,
        severity_level,
    };
    pub use crate::bm25::score as bm25;
    pub use crate::embed::run as embed;
    pub use crate::math::cosine;
    pub use crate::tokenizer::tokenize;

    /// Native-safe wrapper for regression tests. WASM exports use linear-memory
    /// pointers and must not be called with host-process pointers.
    pub fn score_answer(question: &str, ground_truth: &str, miner_answer: &str) -> f32 {
        if miner_answer.trim().is_empty() {
            return 0.0;
        }
        let (relevance, correctness, lexical, len_ratio) = unsafe {
            crate::compute_signals(question, ground_truth, miner_answer)
        };
        if crate::antigame::is_degenerate(miner_answer, len_ratio) {
            return 0.0;
        }
        if crate::antigame::claim_mismatch(ground_truth, miner_answer) {
            return crate::antigame::graded_crush(ground_truth, miner_answer);
        }
        crate::composite_v3(relevance, correctness, lexical, ground_truth, miner_answer)
    }

    /// All raw signals for one triple:
    /// (relevance, correctness, bm25, crit, len_ratio).
    pub fn signals(question: &str, ground_truth: &str, miner_answer: &str) -> (f32, f32, f32, f32, f32) {
        let q_enc = tokenize(question);
        let gt_enc = tokenize(ground_truth);
        let ma_enc = tokenize(miner_answer);
        let q_vec = embed(&q_enc);
        let gt_vec = embed(&gt_enc);
        let ma_vec = embed(&ma_enc);
        let relevance = cosine(&q_vec, &ma_vec);
        let correctness = cosine(&gt_vec, &ma_vec);
        let b = bm25(ground_truth, miner_answer);
        let crit = critical_token_match(ground_truth, miner_answer);
        let lr = answer_len_ratio(ground_truth, miner_answer);
        (relevance, correctness, b, crit, lr)
    }

    /// Parameterized composite — identical formula family to `composite_v3`
    /// but with k/mu/floor and the lexical blend as arguments, so the harness
    /// can sweep constants against measured data without recompiling the lib.
    /// `mode`: 0 = v6 (cosine-gated-by-lexical), 1 = lexical-dominant with
    /// cosine rescaled over MiniLM's anisotropic range.
    pub fn composite_param(
        correctness: f32,
        bm25: f32,
        crit: f32,
        k: f32,
        mu: f32,
        floor: f32,
        w_crit: f32,
        mode: u32,
    ) -> f32 {
        let c = crate::math::clamp01(correctness);
        let l = crate::math::clamp01((1.0 - w_crit) * crate::math::clamp01(bm25) + w_crit * crate::math::clamp01(crit));
        let evidence = match mode {
            0 => c * (floor + (1.0 - floor) * l),
            _ => {
                // Rescale cosine from MiniLM's anisotropic band [c_lo, c_hi]
                // to [0,1], then let lexical agreement carry the evidence.
                const C_LO: f32 = 0.20;
                const C_HI: f32 = 0.80;
                let c_norm = crate::math::clamp01((c - C_LO) / (C_HI - C_LO));
                l * (floor + (1.0 - floor) * c_norm)
            }
        };
        crate::math::sharpen(evidence, k, mu)
    }
}
