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

/// Parse a decimal-like string into an integer representing value * 1000
/// (milli units). Strips trailing zeros and decimal point.
/// Returns None if not a valid decimal number.
fn parse_decimal_milli(s: &str) -> Option<i32> {
    // Quick check: must contain at least one digit
    if !s.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    // Find first digit sequence with optional dot
    let mut num_str = String::new();
    let mut seen_digit = false;
    let mut seen_dot = false;
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            seen_digit = true;
        } else if ch == '.' && !seen_dot && seen_digit {
            num_str.push(ch);
            seen_dot = true;
        } else if seen_digit {
            // Stop at first non-digit after number starts
            break;
        }
    }
    if !seen_digit || num_str.is_empty() {
        return None;
    }
    // Parse to milli: split integer and fractional parts
    let parts: Vec<&str> = num_str.split('.').collect();
    let int_part: i32 = parts[0].parse().ok()?;
    let mut milli = int_part * 1000;
    if parts.len() > 1 {
        let frac = parts[1];
        let frac_len = frac.len().min(3);
        let frac_val: i32 = frac[..frac_len].parse().ok()?;
        // Pad to 3 digits
        let frac_milli = frac_val * 10_i32.pow(3 - frac_len as u32);
        milli += frac_milli;
    }
    Some(milli)
}

/// Check if two decimal strings represent the same numeric value after normalization.
/// Strips trailing zeros and matches exact milli value.
fn numeric_equal(a: &str, b: &str) -> bool {
    parse_decimal_milli(a).is_some_and(|ma| parse_decimal_milli(b).is_some_and(|mb| ma == mb))
}

/// Extract CVE patterns from text and canonicalize: CVE-YYYY-NNNN (zero-padded 4 digits).
/// Handles both "CVE-YYYY-NNNN" and "CVEYYYYNNNN" formats.
fn extract_cves(text: &str) -> Vec<String> {
    let mut cves = Vec::new();
    let upper = text.to_ascii_uppercase();
    let mut search_start = 0;
    while let Some(idx) = upper[search_start..].find("CVE") {
        let abs_idx = search_start + idx;
        let after_cve = &upper[abs_idx + 3..];
        // Check if next char is '-' or digit (start of year)
        let rest = if after_cve.starts_with('-') {
            &after_cve[1..]
        } else if after_cve.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            after_cve
        } else {
            search_start = abs_idx + 4;
            continue;
        };
        // Match exactly 4 digits for year
        if rest.len() < 4 || !rest[..4].chars().all(|c| c.is_ascii_digit()) {
            search_start = abs_idx + 4;
            continue;
        }
        let year = &rest[..4];
        let after_year = &rest[4..];
        // Skip optional dash
        let num_start = if after_year.starts_with('-') { 1 } else { 0 };
        let mut num_end = num_start;
        for (i, ch) in after_year[num_start..].char_indices() {
            if ch.is_ascii_digit() {
                num_end = num_start + i + 1;
            } else {
                break;
            }
        }
        if num_end > num_start {
            let num = &after_year[num_start..num_end];
            let year_digits: String = year.chars().filter(|c| c.is_ascii_digit()).collect();
            let num_digits: String = num.chars().filter(|c| c.is_ascii_digit()).collect();
            if year_digits.len() == 4 && !num_digits.is_empty() {
                let pad_len = 4usize.saturating_sub(num_digits.len());
                let mut num_padded = String::with_capacity(4);
                for _ in 0..pad_len {
                    num_padded.push('0');
                }
                num_padded.push_str(&num_digits);
                let mut canonical = String::with_capacity(13);
                canonical.push_str("CVE-");
                canonical.push_str(&year_digits);
                canonical.push('-');
                canonical.push_str(&num_padded);
                cves.push(canonical);
            }
        }
        search_start = abs_idx + 4;
    }
    cves
}

/// Check if ground-truth CVEs are present in answer.
fn cve_match_score(ground_truth: &str, answer: &str) -> (f32, f32) {
    let gt_cves = extract_cves(ground_truth);
    if gt_cves.is_empty() {
        return (0.0, 0.0); // no CVEs in ground truth, neutral
    }
    let ans_cves = extract_cves(answer);
    let mut hits = 0;
    for cve in &gt_cves {
        if ans_cves.iter().any(|a| a == cve) {
            hits += 1;
        }
    }
    (hits as f32 / gt_cves.len() as f32, gt_cves.len() as f32)
}

/// Anchor keywords that indicate a numeric token is a "fact unit".
const ANCHORS: &[&str] = &["cvss", "cve", "version", "score", "epss", "port", "year", "severity"];

/// Extract fact units from ground truth: (anchor_or_none, numeric_string)
/// Each unit combines a numeric token with its nearest preceding anchor keyword.
fn extract_fact_units(tokens: &[String]) -> Vec<(Option<String>, String)> {
    let mut units = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        // Check if token contains digit
        if tok.chars().any(|c| c.is_ascii_digit()) {
            // Look for anchor in preceding tokens (window 2)
            let mut anchor = None;
            for w in 1..=2 {
                if i >= w {
                    let prev = &tokens[i - w];
                    if ANCHORS.iter().any(|a| prev.contains(a)) {
                        anchor = Some(prev.clone());
                        break;
                    }
                }
            }
            units.push((anchor, tok.clone()));
        }
    }
    units
}

/// Critical-token match upgraded to fact-unit matching with numeric fuzzy and CVE normalization.
/// Returns score in [0,1]: weighted fraction of ground-truth fact units matched in answer.
/// CVE units weigh 2.0, units with anchor keyword weigh 1.0, bare numeric tokens weigh 0.3.
pub fn critical_token_match(ground_truth: &str, answer: &str) -> f32 {
    let gt_tokens = tokenise(ground_truth);
    let ans_tokens = tokenise(answer);
    let ans_set: alloc::collections::BTreeSet<_> = ans_tokens.iter().cloned().collect();

    // CVE matching on raw text (not tokenized)
    let (cve_score, cve_count) = cve_match_score(ground_truth, answer);

    // Fact-unit extraction from tokens (for non-CVE numeric facts)
    let gt_units = extract_fact_units(&gt_tokens);

    let mut total_weight = 0.0f32;
    let mut matched_weight = 0.0f32;

    // Add CVE weight
    if cve_count > 0.0 {
        total_weight += cve_count * 2.0;
        matched_weight += cve_score * cve_count * 2.0;
    }

    for (anchor, num_tok) in gt_units {
        // Skip if this token is part of a CVE (already handled above)
        if num_tok.to_ascii_uppercase().starts_with("CVE") {
            continue;
        }
        let weight = if anchor.is_some() { 1.0 } else { 0.3 };
        total_weight += weight;

        let mut hit = false;
        if let Some(anchor_kw) = anchor {
            // Fact unit: need anchor present AND numeric value match
            // Anchor is a substring match (e.g. "score" matches "scores", "cvss" matches "cvss3")
            let anchor_hit = ans_set.iter().any(|a| a.contains(anchor_kw.as_str()));
            let num_hit = ans_set.iter().any(|a| numeric_equal(a, &num_tok));
            if anchor_hit && num_hit {
                hit = true;
            }
        } else {
            // Bare numeric token: fuzzy numeric match
            if ans_set.iter().any(|a| numeric_equal(a, &num_tok)) {
                hit = true;
            }
        }
        if hit {
            matched_weight += weight;
        }
    }

    if total_weight == 0.0 {
        return 1.0; // neutral, no fact-bearing tokens
    }

    matched_weight / total_weight
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
    fn numeric_equal_strips_trailing_zeros() {
        assert!(numeric_equal("10.0", "10"));
        assert!(numeric_equal("9.8", "9.80"));
        assert!(!numeric_equal("10.0", "9.8"));
    }

    #[test]
    fn cve_canonical_normalization() {
        assert!(cve_match_score("CVE-2021-44228", "cve-2021-44228").0 > 0.9);
        assert!(cve_match_score("CVE-2021-44228", "CVE202144228").0 > 0.9);
        assert!(cve_match_score("CVE-2021-44228", "CVE-2021-44229").0 < 0.1);
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