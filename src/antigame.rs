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
                let len = i - start;
                // Skip single-digit runs: they are almost always embedded in
                // alphanumeric words ("Log4Shell" -> 4, "OpenSSL1.0" -> 1) and
                // are NOT intended factual claims. CVE serial / year / CVSS
                // score / version tokens are all 2+ characters.
                if len < 2 {
                    continue;
                }
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
///
/// `critical_token_match` is the public [0,1] view (figure-less GT -> neutral
/// 1.0). The composite uses `critical_token_match_raw`, which returns the
/// sentinel -1.0 for a figure-less ground truth so the scorer can switch to
/// pure-cosine evidence instead of letting the neutral 1.0 override cosine.
pub fn critical_token_match(ground_truth: &str, answer: &str) -> f32 {
    let m = critical_token_match_raw(ground_truth, answer);
    if m < 0.0 {
        1.0
    } else {
        m
    }
}

/// Sentinel returned when the ground truth carries NO fact-bearing tokens
/// (no CVE ids, no anchored/bare numbers, no versions): -1.0. Callers decide
/// what "no facts to check" means for their composite.
pub fn critical_token_match_raw(ground_truth: &str, answer: &str) -> f32 {
    let lower_gt = ground_truth.to_lowercase();
    let lower_ans = answer.to_lowercase();

    let gt_cve_spans = extract_cve_spans(ground_truth);
    let ans_cve_spans = extract_cve_spans(answer);
    let ans_cves: Vec<String> = ans_cve_spans
        .iter()
        .map(|(c, _, _)| c.clone())
        .collect();

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
            // Paraphrase-tolerant credit: the answer gets credit if it states
            // the GT value ANYWHERE (a correct answer may anchor the number
            // with a different word — GT "severity rating of 10.0" vs answer
            // "CVSS score of 10.0"). We only PENALISE on an active conflict:
            // the answer states a DIFFERENT value near the same anchor stem.
            // This keeps wrong-number answers crushed while letting correct
            // paraphrases through.
            if ans_numbers.iter().any(|(m, _, _, _)| m == milli) {
                matched_weight += W_ANCHORED;
            } else {
                // No exact value anywhere. Check for an active conflict: a
                // different value stated near the same anchor stem in the
                // answer. Score-family anchors are interchangeable.
                let stems: &[&str] = if SCORE_STEMS.contains(&stem) {
                    SCORE_STEMS
                } else {
                    core::slice::from_ref(&stem)
                };
                let mut conflict = false;
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
                            // Digits inside the answer's OWN CVE id must not be
                            // treated as a conflicting figure (the CVE branch
                            // already scores the id itself).
                            if ans_cve_spans.iter().any(|(_, cs, ce)| *ns >= *cs && *ns < *ce) {
                                continue;
                            }
                            if *ns >= lo && *ns < hi && *m != *milli {
                                conflict = true;
                            }
                        }
                        search = occ_end;
                    }
                }
                if conflict {
                    matched_weight -= W_ANCHORED;
                }
            }
        } else {
            total_weight += W_BARE;
            if ans_numbers.iter().any(|(m, _, _, _)| m == milli) {
                matched_weight += W_BARE;
            }
        }
    }

    if total_weight == 0.0 {
        return -1.0; // sentinel: no fact-bearing tokens in GT
    }

    // A CVE identifier alone is an entity key, not a figure-bearing answer
    // lane. When GT contains only a CVE id plus prose (e.g. "affects Log4j2"
    // where the trailing 2 is a single digit embedded in a product name),
    // keep the sentinel. Otherwise a cited CVE would turn lexical coverage
    // into 1.0 and route entity-swap answers through the completeness floor.
    if total_weight == W_CVE {
        return -1.0;
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

/// Count how many of GT's claim figures are missing from the answer.
fn missing_figure_count(ground_truth: &str, answer: &str) -> usize {
    let gt_figs = claim_figures(ground_truth);
    if gt_figs.is_empty() {
        return 0;
    }
    let ans_figs = claim_figures(answer);
    gt_figs.iter().filter(|gf| !ans_figs.contains(gf)).count()
}

/// Graded crush score in (0, CRUSH_SCORE]: an answer missing more of the GT
/// figures crushes lower than one missing fewer. Champion probes follow this
/// shape (only-cve 0.0083, cve+score 0.0146, all-but-sev 0.9995). A fixed
/// CRUSH_SCORE would tie those three against the corresponding good answers,
/// and a tie counts as a loss in the ordering metric.
pub fn graded_crush(ground_truth: &str, answer: &str) -> f32 {
    let gt_figs = claim_figures(ground_truth);
    if gt_figs.is_empty() {
        return CRUSH_SCORE;
    }
    let missing = missing_figure_count(ground_truth, answer);
    let total = gt_figs.len();
    if missing == 0 {
        // No mismatch reason survives here; just return the floor.
        return CRUSH_SCORE;
    }
    // More missing figures → lower score. Anchor: missing=1 of N → ~CRUSH*0.7,
    // missing=total → 0.002. Monotonically decreasing so partial-good vs
    // total-bad always orders correctly.
    let frac_missing = missing as f32 / total as f32;
    let score = CRUSH_SCORE * (1.0 - 0.9 * frac_missing);
    score.max(CRUSH_SCORE * 0.1)
}

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

/// Vulnerability type categories, as a BITMASK. Multiple categories can be
/// present in one text ("remote code execution causing denial of service");
/// the mismatch gate only fires when BOTH sides name exactly ONE category
/// and they differ — anything more ambiguous is left to the semantic path.
fn vuln_type_bits(text: &str) -> u32 {
    // Hyphens normalised to spaces so "denial-of-service" matches the phrase.
    let lower: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c == '-' { ' ' } else { c })
        .collect();

    let mut bits = 0u32;
    // RCE / code execution
    if lower.contains("remote code execution")
        || lower.contains("code execution")
        || lower.contains("arbitrary code")
        || has_word(&lower, "rce")
    {
        bits |= 1;
    }
    // Denial of service
    if lower.contains("denial of service")
        || lower.contains("resource exhaustion")
        || has_word(&lower, "dos")
    {
        bits |= 2;
    }
    // Information disclosure / leak
    if lower.contains("information disclosure")
        || lower.contains("information exposure")
        || lower.contains("information leak")
        || lower.contains("memory disclosure")
        || lower.contains("memory leak")
        || lower.contains("data leak")
        || lower.contains("data exposure")
        || lower.contains("data breach")
        || lower.contains("sensitive data")
    {
        bits |= 4;
    }
    // Injection family
    if lower.contains("sql injection")
        || lower.contains("xpath injection")
        || lower.contains("code injection")
        || lower.contains("command injection")
        || lower.contains("ldap injection")
        || lower.contains("template injection")
    {
        bits |= 8;
    }
    // Privilege escalation / sandbox escape
    if lower.contains("privilege escalation")
        || lower.contains("elevation of privilege")
        || lower.contains("sandbox escape")
        || has_word(&lower, "eop")
    {
        bits |= 16;
    }
    // Authentication / access control
    if lower.contains("authentication bypass")
        || lower.contains("auth bypass")
        || lower.contains("broken auth")
        || lower.contains("access control")
        || has_word(&lower, "csrf")
    {
        bits |= 32;
    }
    // Cross-site scripting
    if lower.contains("cross site scripting") || has_word(&lower, "xss") {
        bits |= 64;
    }
    // Path / directory traversal
    if lower.contains("path traversal") || lower.contains("directory traversal") {
        bits |= 128;
    }
    // SSRF
    if lower.contains("request forgery") || has_word(&lower, "ssrf") {
        bits |= 256;
    }
    // Memory safety
    if lower.contains("buffer overflow")
        || lower.contains("heap overflow")
        || lower.contains("use after free")
        || lower.contains("double free")
        || lower.contains("type confusion")
        || has_word(&lower, "uaf")
    {
        bits |= 512;
    }
    bits
}

/// Whole-word check for short/acronym keywords ("rce", "dos", "xss"): the match
/// must sit on non-alphanumeric boundaries so "kudos" never hits "dos". Works
/// on raw BYTES so multi-byte UTF-8 (emoji, CJK) can never split a char and
/// panic the no_std build.
fn has_word(lower: &str, word: &str) -> bool {
    let bytes = lower.as_bytes();
    let wb = word.as_bytes();
    let wl = wb.len();
    if wl == 0 || bytes.len() < wl {
        return false;
    }
    let mut i = 0usize;
    while i + wl <= bytes.len() {
        if &bytes[i..i + wl] == wb {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_ok = i + wl == bytes.len() || !bytes[i + wl].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Levenshtein edit distance between two short byte strings (bounded: tokens
/// are at most ~24 bytes, so this stays trivially cheap and allocation-free).
fn edit_distance(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut prev: [u8; 33] = [0; 33];
    let mut cur: [u8; 33] = [0; 33];
    if a.len() > 32 || b.len() > 32 {
        return usize::MAX; // out of scope for this gate
    }
    for (j, _) in b.iter().enumerate() {
        prev[j + 1] = j as u8 + 1;
    }
    for i in 1..=a.len() {
        cur[0] = i as u8;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as usize
}

/// Entity near-miss: the answer swaps ONE digit-bearing token for a
/// one-edit-away neighbour — "Log4j2"→"Log4j3", "44228"→"44229", "2021"→"2022"
/// — while omitting the ground-truth token entirely. Cosine and BM25 cannot
/// see a single swapped character; this can. Only fires on digit-bearing
/// tokens (pure-digit runs must be ≥4 chars, mixed tokens ≥3) so
/// plural/inflection differences ("server"/"servers") never trigger it.
fn entity_near_miss(ground_truth: &str, answer: &str) -> bool {
    let gt_tokens = tokenise(ground_truth);
    let ans_tokens = tokenise(answer);

    for t in &gt_tokens {
        let has_digit = t.chars().any(|c| c.is_ascii_digit());
        if !has_digit {
            continue;
        }
        let all_digits = t.chars().all(|c| c.is_ascii_digit());
        if all_digits && t.len() < 4 {
            continue; // years / serials only, not "2", "10"
        }
        if !all_digits && t.len() < 3 {
            continue;
        }
        // Ground-truth token absent from the answer: look for a one-edit
        // impostor the answer uses instead.
        if ans_tokens.iter().any(|u| u == t) {
            continue;
        }
        for u in &ans_tokens {
            if u == t {
                continue;
            }
            let u_digit = u.chars().any(|c| c.is_ascii_digit());
            if !u_digit {
                continue;
            }
            if gt_tokens.iter().any(|g| g == u) {
                continue; // the impostor also appears in GT: not a swap
            }
            if edit_distance(t, u) == 1 {
                return true;
            }
        }
    }
    false
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
/// wrong anchored figure (CVSS/score/year/count under the same anchor), or
/// cites a wrong CVE id. Silence is NOT a contradiction: an answer that names
/// no level/vector/figure/CVE is left to the semantic composite.
///
/// WHY anchored figures: our MiniLM INT8 embedding cannot separate 2021/2017
/// (cos 0.968) or 10.0/7.5 (cos 0.907) — it gives a wrong year/score ~1.0
/// cosine just like a right one. Pure cosine would saturate those bad answers.
/// The lexical anchor-stem match in `critical_token_match` CAN distinguish them
/// because "cvss 10.0" vs "cvss 7.5" are literally different token spans.
/// This gate is the lexical veto on semantic-overlap bad answers.
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

    // Vulnerability type mismatch. Pure cosine cannot separate "denial of
    // service" from "information disclosure" (both sit in the same MiniLM
    // band) — a wrong-type answer keeps all GT figures and rides the cosine.
    // Fires only when both sides name exactly ONE category and they differ.
    let gt_bits = vuln_type_bits(ground_truth);
    let at_bits = vuln_type_bits(answer);
    if gt_bits.count_ones() == 1 && at_bits.count_ones() == 1 && gt_bits != at_bits {
        return true;
    }

    // Entity near-miss: a digit-bearing GT token swapped for a one-edit
    // impostor ("Log4j2"→"Log4j3") that appears nowhere in the ground truth.
    if entity_near_miss(ground_truth, answer) {
        return true;
    }

    // Wrong CVE id: ground truth names a CVE and the answer cites a DIFFERENT
    // CVE. Pure cosine can't tell CVE-123 from CVE-456, but they are distinct
    // facts — crush it.
    let (gt_cve_hits, gt_cve_count) = cve_match_score(ground_truth, answer);
    if gt_cve_count > 0.0 && gt_cve_hits == 0.0 {
        let ans_has_any = !extract_cve_spans(answer).is_empty();
        if ans_has_any {
            return true;
        }
    }

    // Anchored numeric mismatch is handled by the claim_figures gate below:
    // GT states figures, the answer states figures, and none of the answer's
    // figures match any of GT's → wrong year/score/count → crush.

    // Completeness gate (champion behavior): if the ground truth states figures,
    // the answer MUST cover ALL of them. Missing a figure -> crushed.
    // Champion probes: only-cve 0.0083, cve+score 0.0146 (crushed), but
    // all-figures-no-severity 0.9995 (rides). So figures are required, severity
    // word is NOT — wrong severity is caught by the mismatch gate above.
    let gt_figs = claim_figures(ground_truth);
    let ans_figs = claim_figures(answer);
    if !gt_figs.is_empty() && !gt_figs.iter().all(|gf| ans_figs.contains(gf)) {
        return true;
    }

    // Extra-figure gate: if the answer introduces a numeric claim that does
    // NOT appear in the ground truth AND that ground truth already states
    // figures, treat it as a confident-but-wrong fabrication (e.g. "affecting
    // over 100,000 servers"). Champion probes (extra figure vs correct) → crush.
    // Excludes bare CVE-serial digits and version numbers which are secondary
    // facts, and skips GT figures themselves (they're covered by completeness).
    if !gt_figs.is_empty() {
        for af in &ans_figs {
            if gt_figs.contains(af) {
                continue; // matches a GT figure, allowed
            }
            // Single-digit tokens are still allowed if not already filtered
            // at the extract layer (we skipped len<2 there).
            // Tolerance: very small numeric additions that read as ordinals
            // (1, 2, 3...) rarely qualify as confident claims — let them
            // through unless they're already flagged as a CVE-internal serial.
            // Only flag distinctive multi-digit figures (>= 100) which read as
            // counts, percentages, or external scale claims.
            if af.abs() >= 100 {
                return true;
            }
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