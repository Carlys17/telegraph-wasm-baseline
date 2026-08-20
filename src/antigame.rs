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

/// Critical-token match: fraction of the ground truth's *fact-bearing* tokens
/// that appear verbatim in the answer.
///
/// For factual intents (CVE_LOOKUP, ONCHAIN_TX_LOOKUP, SSL_VERIFICATION…) the
/// answer is right or wrong on a handful of hard tokens: a CVSS score (`9.8`),
/// a severity word, a version string, a CVE id. MiniLM cosine is notoriously
/// insensitive to exactly these — "CVSS 9.8 critical" and "CVSS 7.5 high" embed
/// almost identically — so cosine + BM25 alone cannot separate a correct answer
/// from a topically-identical wrong-number answer. This signal can: it keys on
/// tokens that carry a digit (scores, versions, ids, dates), which is where the
/// factual truth lives.
///
/// A ground truth with no digit-bearing tokens falls back to 1.0 (neutral — this
/// signal has nothing to say, let cosine/BM25 decide).
pub fn critical_token_match(ground_truth: &str, answer: &str) -> f32 {
    let gt_terms = tokenise(ground_truth);
    let ans_terms = tokenise(answer);

    // Fact-bearing = contains at least one ASCII digit.
    let mut critical: Vec<&str> = Vec::new();
    for t in &gt_terms {
        if t.chars().any(|c| c.is_ascii_digit()) && !critical.iter().any(|s| *s == t.as_str()) {
            critical.push(t.as_str());
        }
    }

    if critical.is_empty() {
        return 1.0;
    }

    let mut hits = 0u32;
    for c in &critical {
        if ans_terms.iter().any(|a| a.as_str() == *c) {
            hits += 1;
        }
    }

    hits as f32 / critical.len() as f32
}

/// Distinct-token ratio: unique terms / total terms in the answer. Used only by
/// the degenerate-answer gate below, not as a smooth score multiplier.
fn distinct_ratio(answer: &str) -> f32 {
    let terms = tokenise(answer);
    let total = terms.len();
    if total < 4 {
        return 1.0;
    }

    let mut seen: Vec<&str> = Vec::new();
    for t in &terms {
        if !seen.iter().any(|s| *s == t.as_str()) {
            seen.push(t.as_str());
        }
    }

    seen.len() as f32 / total as f32
}

/// Hard gate for degenerate answers, returning `true` when the answer should be
/// forced to 0 *before* any scoring.
///
/// This replaces v2's smooth multiplicative `stuffing_penalty`, which was the
/// design mistake that lost on separation: multiplying the final composite by a
/// penalty in [0.2,1.0] pulled every score toward the middle and collapsed the
/// good/bad gap. Gaming defence belongs at the extremes — a clearly-degenerate
/// answer is slammed to exactly 0 (which *helps* separation), and everything
/// else passes through untouched so the sharpening transform can do its job.
pub fn is_degenerate(answer: &str, answer_len_ratio: f32) -> bool {
    // Empty / whitespace-only, or answer far too short to be a real answer.
    if answer.trim().is_empty() || answer_len_ratio < EMPTY_RATIO {
        return true;
    }
    // Massive padding relative to the ground truth = length farming.
    if answer_len_ratio > STUFF_RATIO {
        return true;
    }
    // Crude single-token spam: many tokens, almost all identical.
    let terms = tokenise(answer);
    if terms.len() >= 8 && distinct_ratio(answer) < 0.20 {
        return true;
    }
    false
}

/// Answer length ratio = answer token count / ground-truth token count.
/// Feeds the degenerate gate (empty / padding detection). Returns a large
/// value when the ground truth has no tokens but the answer does (so an answer
/// against an empty reference isn't falsely gated as "too short").
pub fn answer_len_ratio(ground_truth: &str, answer: &str) -> f32 {
    let gt_len = tokenise(ground_truth).len() as f32;
    let ans_len = tokenise(answer).len() as f32;
    if gt_len == 0.0 {
        return if ans_len == 0.0 { 0.0 } else { 1.0 };
    }
    ans_len / gt_len
}

/// Answer shorter than 3% of the ground truth's token count → treat as empty.
const EMPTY_RATIO: f32 = 0.03;
/// Answer longer than 8× the ground truth → treat as padding/farming.
const STUFF_RATIO: f32 = 8.0;

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_match_exact_numbers() {
        let gt = "CVE-2021-44228 has CVSS score 10.0 and critical severity";
        let good = "The CVE-2021-44228 vulnerability scores 10.0 CVSS, critical.";
        let m = critical_token_match(gt, good);
        assert!(m > 0.9, "correct numbers should match, got {m:.3}");
    }

    #[test]
    fn critical_match_wrong_numbers_low() {
        let gt = "CVSS score 10.0 for 2021";
        let wrong = "CVSS score 7.5 for 2019"; // topically identical, wrong facts
        let m = critical_token_match(gt, wrong);
        assert!(m < 0.5, "wrong numbers should score low, got {m:.3}");
    }

    #[test]
    fn critical_match_no_digits_neutral() {
        assert_eq!(critical_token_match("critical severity remote code", "anything here"), 1.0);
    }

    #[test]
    fn degenerate_empty() {
        assert!(is_degenerate("", 0.0));
        assert!(is_degenerate("   ", 0.0));
    }

    #[test]
    fn degenerate_padding() {
        assert!(is_degenerate("some real words here padded out", 12.0));
    }

    #[test]
    fn degenerate_single_word_spam() {
        assert!(is_degenerate("paris paris paris paris paris paris paris paris", 1.0));
    }

    #[test]
    fn healthy_answer_not_degenerate() {
        assert!(!is_degenerate(
            "CVE-2021-44228 is a critical remote code execution flaw in Log4j.",
            1.1
        ));
    }
}
