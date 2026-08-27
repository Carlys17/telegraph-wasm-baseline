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

/// Anchor STEMS indicating a nearby numeric token is a "fact unit". Stems so
/// "score"/"scores"/"scoring" all match "scor", "version"/"versions" match
/// "vers", etc.
const ANCHOR_STEMS: &[&str] = &["cvss", "scor", "epss", "port", "vers", "sever", "base"];

/// Score-family stems: GT "base score of 9.8" and answer "CVSS 9.8" name the
/// same fact with different anchor words, so when the GT fact's anchor is one
/// of these, the answer's values are collected near ANY of them.
const SCORE_STEMS: &[&str] = &["cvss", "scor", "epss", "base"];

/// Weight per ground-truth CVE id.
const W_CVE: f32 = 1.0;
/// Weight per anchored numeric fact — the numbers that carry the truth
/// (CVSS score, version, port...). Dominates the crit signal on purpose:
/// CVE_LOOKUP answers live or die by their numbers.
const W_ANCHORED: f32 = 3.0;
/// Weight per bare numeric token (year, count, ...) with no anchor nearby.
const W_BARE: f32 = 0.3;
/// Weight per version number ("v3.1", "SMBv1"). Versions are secondary facts;
/// they never trigger a conflict penalty (a correct answer that says
/// "CVSS 10.0" next to "CVSS v3.1" must not be punished for the 3.1).
const W_VERSION: f32 = 0.5;
/// How many bytes around a number/anchor we scan for the (number, anchor)
/// association, in raw text.
const FACT_WINDOW: usize = 40;

/// Small number words accepted as numeric values (answers sometimes spell
/// scores out: "a score of ten"). Milli units (value × 1000).
const WORD_NUMBERS: &[(&str, i32)] = &[
    ("zero", 0),
    ("one", 1_000),
    ("two", 2_000),
    ("three", 3_000),
    ("four", 4_000),
    ("five", 5_000),
    ("six", 6_000),
    ("seven", 7_000),
    ("eight", 8_000),
    ("nine", 9_000),
    ("ten", 10_000),
];

/// Snap a byte offset to a char boundary (floor) so slicing never panics.
fn snap(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Extract decimal numbers from raw text WITH byte spans:
/// (milli value, start, end, is_version). Operates on raw text, not the
/// alphanumeric tokeniser — that one destroys decimals ("7.5" → "7","5",
/// dropped by the len>=2 filter), which was the v6 bug that let wrong-score
/// answers score crit=1.0.
///
/// `is_version` = the digit run is immediately preceded by 'v'/'V'
/// ("v3.1", "SMBv1"). Version digits are secondary facts, weighted low, and
/// never conflict-penalised.
fn extract_numbers_raw(text: &str) -> Vec<(i32, usize, usize, bool)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut seen_dot = false;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || (bytes[i] == b'.'
                        && !seen_dot
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()))
            {
                if bytes[i] == b'.' {
                    seen_dot = true;
                }
                i += 1;
            }
            if i > start {
                if let Some(milli) = parse_decimal_milli(&text[start..i]) {
                    let is_version = start > 0
                        && (bytes[start - 1] == b'v' || bytes[start - 1] == b'V');
                    out.push((milli, start, i, is_version));
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Number words ("ten", "seven"...) as (milli, start, end), whole-word only.
fn word_numbers(text: &str) -> Vec<(i32, usize, usize)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    for (word, milli) in WORD_NUMBERS {
        let wl = word.len();
        let mut i = 0;
        while i + wl <= bytes.len() {
            let mut found = false;
            if let Some(slice) = text.get(i..i + wl) {
                if slice.eq_ignore_ascii_case(word) {
                    let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                    let after_ok = i + wl == bytes.len() || !bytes[i + wl].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        out.push((*milli, i, i + wl));
                        found = true;
                    }
                }
            }
            i += if found { wl } else { 1 };
        }
    }
    out
}

/// All numeric values in text: digit runs (with version flag) + number words.
fn all_numbers(text: &str) -> Vec<(i32, usize, usize, bool)> {
    let mut v = extract_numbers_raw(text);
    for (m, s, e) in word_numbers(text) {
        v.push((m, s, e, false));
    }
    v.sort_by_key(|t| t.1);
    v
}

/// Extract CVE ids with byte spans: (canonical id, span_start, span_end).
fn extract_cve_spans(text: &str) -> Vec<(String, usize, usize)> {
    let mut cves = Vec::new();
    let upper = text.to_ascii_uppercase();
    let mut search_start = 0;
    while let Some(idx) = upper[search_start..].find("CVE") {
        let abs_idx = search_start + idx;
        let after_cve = &upper[abs_idx + 3..];
        let dash = after_cve.starts_with('-');
        let rest = if dash {
            &after_cve[1..]
        } else if after_cve.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            after_cve
        } else {
            search_start = abs_idx + 4;
            continue;
        };
        if rest.len() < 4 || !rest[..4].chars().all(|c| c.is_ascii_digit()) {
            search_start = abs_idx + 4;
            continue;
        }
        let year = &rest[..4];
        let after_year = &rest[4..];
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
                let rest_abs = abs_idx + 3 + if dash { 1 } else { 0 };
                let end_abs = rest_abs + 4 + num_end;
                cves.push((canonical, abs_idx, end_abs));
            }
        }
        search_start = abs_idx + 4;
    }
    cves
}

/// Check if ground-truth CVEs are present in answer.
fn cve_match_score(ground_truth: &str, answer: &str) -> (f32, f32) {
    let gt_cves = extract_cve_spans(ground_truth);
    if gt_cves.is_empty() {
        return (0.0, 0.0); // no CVEs in ground truth, neutral
    }
    let ans_cves: Vec<String> = extract_cve_spans(answer).into_iter().map(|(c, _, _)| c).collect();
    let mut hits = 0;
    for (c, _, _) in &gt_cves {
        if ans_cves.iter().any(|a| a == c) {
            hits += 1;
        }
    }
    (hits as f32 / gt_cves.len() as f32, gt_cves.len() as f32)
}

/// Critical-token match v2: raw-text fact-unit matching with numeric fuzzy
/// equality, CVE normalization, and CONFLICT PENALTY.
///
/// Returns score in [0,1]: weighted fraction of ground-truth fact units the
/// answer gets right, minus penalties for actively contradicting a fact
/// (e.g. GT "score of 7.5" vs answer "score of 9.8" near the same anchor
/// subtracts instead of merely missing). This is what separates a
/// topically-identical WRONG answer from a correct one: cosine and BM25 can't
/// tell 7.5 from 9.8, this signal can.
///
/// Weights: CVE id 1.0, anchored number 3.0, bare number 0.3.
pub fn critical_token_match(ground_truth: &str, answer: &str) -> f32 {
    let lower_gt = ground_truth.to_lowercase();
    let lower_ans = answer.to_lowercase();

    let gt_cve_spans = extract_cve_spans(ground_truth);
    let ans_cves: Vec<String> = extract_cve_spans(answer).into_iter().map(|(c, _, _)| c).collect();

    let mut total_weight = 0.0f32;
    let mut matched_weight = 0.0f32;

    // ── CVE ids ──────────────────────────────────────────────────────────────
    // For CVE_LOOKUP the CVE id is already in the QUESTION, so a correct
    // answer may legitimately omit it. The id only becomes a scored fact when
    // the answer cites a CVE: right id = credit, wrong id = penalty.
    if !gt_cve_spans.is_empty() && !ans_cves.is_empty() {
        let mut hits = 0usize;
        for (c, _, _) in &gt_cve_spans {
            if ans_cves.iter().any(|a| a == c) {
                hits += 1;
            }
        }
        total_weight += gt_cve_spans.len() as f32 * W_CVE;
        matched_weight += hits as f32 * W_CVE;
        // Answer cites a CVE but none of the ground truth's: actively wrong id.
        if hits == 0 {
            matched_weight -= W_CVE;
        }
    }

    // ── Numeric facts (raw text, decimals + number words preserved) ─────────
    let gt_numbers = all_numbers(ground_truth);
    let ans_numbers = all_numbers(answer);

    for (milli, start, end, is_version) in &gt_numbers {
        // Digits inside a CVE id are handled by the CVE branch above.
        if gt_cve_spans.iter().any(|(_, cs, ce)| *start >= *cs && *end <= *ce) {
            continue;
        }

        // Version numbers ("v3.1", "SMBv1"): low weight, match-only, never
        // conflict-penalised — a correct answer that omits or rephrases the
        // version must not be punished.
        if *is_version {
            total_weight += W_VERSION;
            if ans_numbers.iter().any(|(m, _, _, _)| m == milli) {
                matched_weight += W_VERSION;
            }
            continue;
        }

        // Nearest anchor stem within FACT_WINDOW bytes BEFORE or AFTER the
        // number (answers say both "CVSS 10.0" and "10.0 CVSS").
        let win_start = snap(&lower_gt, start.saturating_sub(FACT_WINDOW));
        let win_end = snap(&lower_gt, (*end + FACT_WINDOW).min(lower_gt.len()));
        let window = &lower_gt[win_start..win_end];
        let num_rel = *start - win_start;
        let num_len = *end - *start;
        let mut anchor: Option<&'static str> = None;
        let mut best_dist = usize::MAX;
        for stem in ANCHOR_STEMS {
            let mut search = 0usize;
            while search < window.len() {
                let rel = match window[search..].find(stem) {
                    Some(p) => p,
                    None => break,
                };
                let occ = search + rel;
                let dist = if occ + stem.len() <= num_rel {
                    num_rel - (occ + stem.len())
                } else if occ >= num_rel + num_len {
                    occ - (num_rel + num_len)
                } else {
                    0
                };
                if dist < best_dist {
                    best_dist = dist;
                    anchor = Some(stem);
                }
                search = occ + stem.len();
            }
        }

        if let Some(stem) = anchor {
            total_weight += W_ANCHORED;
            // Values the answer states near the SAME anchor stem (either side).
            // Score-family anchors are interchangeable: GT "base score of 9.8"
            // vs answer "CVSS 9.8" is the same fact.
            let stems: &[&str] = if SCORE_STEMS.contains(&stem) {
                SCORE_STEMS
            } else {
                core::slice::from_ref(&stem)
            };
            let mut values: Vec<i32> = Vec::new();
            for st in stems {
                let mut search = 0usize;
                while search < lower_ans.len() {
                    let rel = match lower_ans[search..].find(st) {
                        Some(p) => p,
                        None => break,
                    };
                    let occ_start = search + rel;
                    let occ_end = occ_start + st.len();
                    let lo = snap(&lower_ans, occ_start.saturating_sub(FACT_WINDOW));
                    let hi = snap(&lower_ans, (occ_end + FACT_WINDOW).min(lower_ans.len()));
                    for (m, ns, _, _) in &ans_numbers {
                        if *ns >= lo && *ns < hi {
                            values.push(*m);
                        }
                    }
                    search = occ_end;
                }
            }
            if values.iter().any(|v| v == milli) {
                matched_weight += W_ANCHORED;
            } else if !values.is_empty() {
                // Conflicting value next to the same anchor: worse than silence.
                matched_weight -= W_ANCHORED;
            }
        } else {
            total_weight += W_BARE;
            if ans_numbers.iter().any(|(m, _, _, _)| m == milli) {
                matched_weight += W_BARE;
            }
        }
    }

    if total_weight == 0.0 {
        return 1.0; // neutral, no fact-bearing tokens
    }

    (matched_weight / total_weight).clamp(0.0, 1.0)
}

// ── Categorical claim gates (v14 hybrid) ────────────────────────────────────
// Cosine + BM25 cannot tell "critical" from "high" or "network" from "local":
// a wrong-but-plausible categorical answer sits in the same embedding band as
// a correct one and saturates the composite to ~1.0. These gates detect the
// answer actively contradicting a categorical fact the ground truth states,
// so the composite can crush it instead of crediting its topical similarity.

/// Hard crush score for a confidently wrong categorical claim. A constant
/// (not derived) so the WASM binary stays byte-reproducible across builds.
pub const CRUSH_SCORE: f32 = 0.02;

/// CVSS severity level named in `text`: 0 none, 1 low, 2 medium, 3 high,
/// 4 critical. First level word wins (answers lead with the rating).
pub fn severity_level(text: &str) -> u32 {
    for t in tokenise(text) {
        match t.as_str() {
            "critical" => return 4,
            "high" => return 3,
            "medium" | "moderate" => return 2,
            "low" => return 1,
            _ => {}
        }
    }
    0
}

/// CVSS attack vector named in `text`: 0 none, 1 network, 2 adjacent,
/// 3 local, 4 physical. First vector word wins — a wrong answer often names
/// several ("adjacent ... local network position") and the leading one is the
/// claim being made.
pub fn attack_vector(text: &str) -> u32 {
    for t in tokenise(text) {
        match t.as_str() {
            "network" => return 1,
            "adjacent" => return 2,
            "local" => return 3,
            "physical" => return 4,
            _ => {}
        }
    }
    0
}

/// Figures that carry a claim: all numbers in `text` EXCEPT digits inside CVE
/// id spans (the year/serial of "CVE-2021-44228" are identifiers, not claimed
/// values — without this exclusion an answer citing the right CVE id would
/// "satisfy" a ground-truth year of 2021 while asserting disclosure in 2017)
/// and EXCEPT version numbers ("v3.1", secondary facts, never conflict-worthy).
fn claim_figures(text: &str) -> Vec<i32> {
    let spans = extract_cve_spans(text);
    all_numbers(text)
        .into_iter()
        .filter(|(_m, s, e, is_version)| {
            !*is_version && !spans.iter().any(|(_, cs, ce)| *s >= *cs && *e <= *ce)
        })
        .map(|(m, _, _, _)| m)
        .collect()
}

/// True when the answer actively contradicts a categorical fact the ground
/// truth states: a different severity level, a different attack vector, a
/// confident wrong figure (GT states figures, the answer misses every one of
/// them yet states figures of its own — wrong year, wrong count), or cites a
/// wrong CVE id. Silence is NOT a contradiction: an answer that names no
/// level/vector/figure/CVE is left to the semantic composite.
pub fn claim_mismatch(ground_truth: &str, answer: &str) -> bool {
    let gs = severity_level(ground_truth);
    let asv = severity_level(answer);
    if gs != 0 && asv != 0 && gs != asv {
        return true;
    }

    let gv = attack_vector(ground_truth);
    let av = attack_vector(answer);
    if gv != 0 && av != 0 && gv != av {
        return true;
    }

    // Wrong CVE id: ground truth names a CVE and the answer cites a DIFFERENT
    // CVE. Pure cosine can't tell CVE-123 from CVE-456, but they are distinct
    // facts — crush it.
    let (gt_cve_hits, gt_cve_count) = cve_match_score(ground_truth, answer);
    if gt_cve_count > 0.0 && gt_cve_hits == 0.0 {
        // ground truth has CVEs; answer either cites none or none match.
        let ans_has_any = !extract_cve_spans(answer).is_empty();
        if ans_has_any {
            return true; // answer cites a CVE but the wrong one(s)
        }
    }

    let gt_figs = claim_figures(ground_truth);
    if !gt_figs.is_empty() {
        let ans_figs = claim_figures(answer);
        if !ans_figs.is_empty() && !gt_figs.iter().any(|m| ans_figs.contains(m)) {
            return true;
        }
    }

    false
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