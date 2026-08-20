//! Gaming-resistance signals — the core of this evaluator's edge over the
//! Telegraph baseline.
//!
//! The stock baseline (`rank_answer`) has three exploitable weaknesses that
//! let a low-quality miner inflate its score without improving real answer
//! quality:
//!
//!   1. **Keyword stuffing** — BM25 rewards ground-truth term overlap, but
//!      never punishes *repeating* those terms. An answer that spams the
//!      ground-truth keywords over and over scores high lexical overlap while
//!      being unreadable garbage.
//!   2. **Padding / length farming** — the length signal is
//!      `sigmoid((len-50)/20)`: a pure function of raw byte length. Appending
//!      50 chars of filler is free points regardless of relevance.
//!   3. **Copy-the-question** — an answer that just echoes the question (or
//!      the ground truth verbatim) rides cosine similarity without adding
//!      information.
//!
//! This module produces two multiplicative penalty factors in [0,1] that the
//! composite applies on top of the base weighted sum, so a gamed answer gets
//! its inflated base score pulled back down. Both are deterministic and
//! no_std (Vec/String only), matching the rest of the pipeline.

extern crate alloc;

use alloc::{string::String, vec::Vec};

/// Tokenise into lowercase alphanumeric words, min length 2 — identical rules
/// to `bm25::tokenise` so the two signals see the same token stream.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| {
            s.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        (c as u8 + 32) as char
                    } else {
                        c
                    }
                })
                .collect()
        })
        .collect()
}

/// Distinct-token ratio: unique terms / total terms in the answer.
///
/// A natural human/model answer has high lexical diversity (ratio near 1 for
/// short text, settling ~0.4–0.7 for longer prose). A keyword-stuffed answer
/// that repeats the same handful of ground-truth terms collapses toward a low
/// ratio. Returns 1.0 for trivially short answers (< 4 tokens) where a low
/// ratio isn't evidence of gaming.
pub fn distinct_ratio(answer: &str) -> f32 {
    let terms = tokenise(answer);
    let total = terms.len();
    if total < 4 {
        return 1.0;
    }

    // Count unique terms — linear scan is fine for realistic answer lengths.
    let mut seen: Vec<&str> = Vec::new();
    for t in &terms {
        if !seen.iter().any(|s| *s == t.as_str()) {
            seen.push(t.as_str());
        }
    }

    seen.len() as f32 / total as f32
}

/// Max single-term frequency share: how much of the answer is one repeated word.
///
/// Catches the crude stuffing attack where a miner pastes one high-value
/// ground-truth keyword dozens of times. Returns the fraction the most common
/// term occupies (0..1). A healthy answer sits low (~0.1–0.2); a spam answer
/// spikes toward 1.0.
fn max_term_share(answer: &str) -> f32 {
    let terms = tokenise(answer);
    let total = terms.len();
    if total < 4 {
        return 0.0;
    }

    let mut counts: Vec<(&str, u32)> = Vec::new();
    for t in &terms {
        if let Some(e) = counts.iter_mut().find(|(s, _)| *s == t.as_str()) {
            e.1 += 1;
        } else {
            counts.push((t.as_str(), 1));
        }
    }

    let max = counts.iter().map(|(_, c)| *c).max().unwrap_or(0);
    max as f32 / total as f32
}

/// Combined stuffing penalty in [0,1] (1.0 = no penalty, clean answer).
///
/// Blends distinct-ratio and max-term-share so both the "repeat many keywords"
/// and "repeat one keyword" variants are caught. The penalty is intentionally
/// forgiving in the healthy region (diversity above ~0.6 barely dents the
/// score) and bites hard only when an answer is clearly degenerate.
pub fn stuffing_penalty(answer: &str) -> f32 {
    let diversity = distinct_ratio(answer);          // high = good
    let dominance = max_term_share(answer);          // high = bad

    // diversity_factor: maps diversity [0,1] → penalty. Full credit at ≥0.6,
    // linear falloff below, floor at 0.2 so a gamed answer keeps at most 20%.
    let diversity_factor = if diversity >= 0.6 {
        1.0
    } else {
        0.2 + (diversity / 0.6) * 0.8
    };

    // dominance_factor: one word owning >30% of the answer starts to hurt.
    let dominance_factor = if dominance <= 0.30 {
        1.0
    } else {
        // 0.30 → 1.0, 1.0 → 0.2, linear
        let over = (dominance - 0.30) / 0.70;
        crate::math::clamp01(1.0 - over * 0.8)
    };

    crate::math::clamp01(diversity_factor * dominance_factor)
}

/// Length-quality relative to the ground truth, replacing the baseline's
/// absolute `sigmoid((len-50)/20)`.
///
/// The baseline rewards raw length: a 400-char padded answer beats a tight
/// 60-char correct one. Instead we score length by how close the answer's
/// word count is to the ground truth's, on a log-ratio Gaussian that peaks at
/// parity (r = 1.0), tolerates answers modestly longer or shorter, and
/// penalises both terse stubs and bloated padding. Ground truth with no
/// tokens falls back to a mild neutral score.
pub fn relative_length_quality(ground_truth: &str, answer: &str) -> f32 {
    let gt_len = tokenise(ground_truth).len() as f32;
    let ans_len = tokenise(answer).len() as f32;

    if ans_len == 0.0 {
        return 0.0;
    }
    if gt_len == 0.0 {
        // No reference length to compare against — give a neutral, capped score
        // that still rewards having *some* content without length farming.
        return crate::math::clamp01(ans_len / (ans_len + 8.0));
    }

    // log-ratio via libm; peak at ratio 1.0. sigma controls tolerance: at
    // sigma=0.9, a 2x-longer or 2x-shorter answer (|ln 2|≈0.69) keeps ~78%.
    let ratio = ans_len / gt_len;
    let log_r = libm::logf(ratio);
    const SIGMA: f32 = 0.9;
    let q = libm::expf(-(log_r * log_r) / (2.0 * SIGMA * SIGMA));
    crate::math::clamp01(q)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_answer_no_stuffing_penalty() {
        let p = stuffing_penalty(
            "Paris is the capital of France and its largest city by population.",
        );
        assert!(p > 0.9, "clean prose should barely be penalised, got {p:.3}");
    }

    #[test]
    fn repeated_single_keyword_heavily_penalised() {
        let p = stuffing_penalty("paris paris paris paris paris paris paris paris");
        assert!(p < 0.4, "single-word spam should be penalised, got {p:.3}");
    }

    #[test]
    fn keyword_list_stuffing_penalised() {
        let p = stuffing_penalty("france paris france paris france paris france paris");
        assert!(p < 0.6, "two-word spam should be penalised, got {p:.3}");
    }

    #[test]
    fn short_answer_not_penalised() {
        let p = stuffing_penalty("yes");
        assert_eq!(p, 1.0);
    }

    #[test]
    fn length_peaks_at_parity() {
        let gt = "the capital of france is paris";
        let same = relative_length_quality(gt, "the capital city of france is paris");
        let padded = relative_length_quality(
            gt,
            "the capital of france is paris and also here is a lot of extra filler text \
             that keeps going and going well beyond what the ground truth ever needed",
        );
        assert!(same > padded, "parity ({same:.3}) should beat padding ({padded:.3})");
    }

    #[test]
    fn empty_answer_zero_length_quality() {
        assert_eq!(relative_length_quality("some ground truth", ""), 0.0);
    }
}
