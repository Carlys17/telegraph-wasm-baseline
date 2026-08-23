//! Native measurement harness — runs the crate's REAL MiniLM embedder +
//! scoring internals on realistic CVE_LOOKUP triples and reports the actual
//! signal operating points (cosines, BM25, critical-token match) plus the
//! resulting separation margin for candidate constant sets.
//!
//! Run:
//!   cargo test --features "real_weights bench" --test realbench -- --nocapture
//!
//! This exists because the previous `tools/sep_bench.rs` GUESSED cosine
//! operating points (good≈0.85-0.95). Real MiniLM-L6-v2 paraphrase cosines are
//! much lower due to anisotropy, which is why every submission undershot the
//! champion by a constant ~0.09. Here we MEASURE instead of guess.

use telegraph_scoring::bench_api as B;

/// One evaluation triple with a class label.
struct Case {
    label: &'static str,
    question: &'static str,
    ground_truth: &'static str,
    answer: &'static str,
    good: bool,
}

fn sharpen(raw: f32, k: f32, mu: f32) -> f32 {
    let x = (-k * (raw - mu)).exp();
    1.0 / (1.0 + x)
}
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// v6 composite (mode 0): cosine-gated-by-lexical.
fn v6(c: f32, bm25: f32, crit: f32, k: f32, mu: f32, floor: f32, w_crit: f32) -> f32 {
    let l = clamp01((1.0 - w_crit) * clamp01(bm25) + w_crit * clamp01(crit));
    let evidence = clamp01(c) * (floor + (1.0 - floor) * l);
    sharpen(evidence, k, mu)
}

/// v7 candidate (mode 1): rescale cosine over MiniLM's anisotropic band to
/// [0,1], then let lexical agreement carry the evidence.
fn v7(c: f32, bm25: f32, crit: f32, k: f32, mu: f32, floor: f32, w_crit: f32, c_lo: f32, c_hi: f32) -> f32 {
    let l = clamp01((1.0 - w_crit) * clamp01(bm25) + w_crit * clamp01(crit));
    let c_norm = clamp01((clamp01(c) - c_lo) / (c_hi - c_lo));
    let evidence = l * (floor + (1.0 - floor) * c_norm);
    sharpen(evidence, k, mu)
}

fn main() {
    let cases = build_cases();
    println!("=== measuring {} triples with REAL MiniLM ===\n", cases.len());

    // Measure raw signals for every case.
    let mut rows: Vec<(bool, f32, f32, f32, f32, f32, &'static str)> = Vec::new();
    for c in &cases {
        let (rel, corr, bm25, crit, lr) = B::signals(c.question, c.ground_truth, c.answer);
        let deg = B::is_degenerate(c.answer, lr);
        rows.push((c.good, corr, bm25, crit, lr, rel, c.label));
        println!(
            "{:4} {:22} corr={:.3} bm25={:.3} crit={:.3} lenr={:.2} rel={:.3}{}",
            if c.good { "GOOD" } else { "BAD " },
            c.label,
            corr,
            bm25,
            crit,
            lr,
            rel,
            if deg { "  [DEGEN->0]" } else { "" }
        );
    }

    // Operating-point summary per class.
    let good: Vec<_> = rows.iter().filter(|r| r.0).collect();
    let bad: Vec<_> = rows.iter().filter(|r| !r.0).collect();
    let mean = |v: &Vec<&(bool, f32, f32, f32, f32, f32, &'static str)>, f: fn(&&(bool, f32, f32, f32, f32, f32, &'static str)) -> f32| {
        v.iter().map(f).sum::<f32>() / v.len() as f32
    };
    println!("\n=== operating points (measured) ===");
    println!(
        "GOOD(n={}): corr mean={:.3} min={:.3} | bm25 mean={:.3} | crit mean={:.3}",
        good.len(),
        mean(&good, |r| r.1),
        good.iter().map(|r| r.1).fold(9.0f32, f32::min),
        mean(&good, |r| r.2),
        mean(&good, |r| r.3)
    );
    println!(
        "BAD (n={}): corr mean={:.3} max={:.3} | bm25 mean={:.3} | crit mean={:.3}",
        bad.len(),
        mean(&bad, |r| r.1),
        bad.iter().map(|r| r.1).fold(-9.0f32, f32::max),
        mean(&bad, |r| r.2),
        mean(&bad, |r| r.3)
    );

    // Sweep candidate constant sets and report margins.
    println!("\n=== margin sweep (mean(good)-mean(bad)) ===");
    let configs: &[(&str, f32, f32, f32, f32)] = &[
        // (name, k, mu, floor, w_crit)
        ("v6-shipped k22 mu.52 fl.12 wc.80", 22.0, 0.52, 0.12, 0.80),
        ("k22 mu.45 fl.12 wc.80", 22.0, 0.45, 0.12, 0.80),
        ("k22 mu.40 fl.12 wc.80", 22.0, 0.40, 0.12, 0.80),
        ("k26 mu.45 fl.10 wc.85", 26.0, 0.45, 0.10, 0.85),
        ("k30 mu.40 fl.10 wc.90", 30.0, 0.40, 0.10, 0.90),
    ];
    for (name, k, mu, floor, wc) in configs {
        let mg = score_class(&rows, true, |c, b, cr| v6(c, b, cr, *k, *mu, *floor, *wc));
        let mb = score_class(&rows, false, |c, b, cr| v6(c, b, cr, *k, *mu, *floor, *wc));
        println!("mode0 {:34} good={:.4} bad={:.4} margin={:.4}", name, mg, mb, mg - mb);
    }
    // mode1 (cosine-rescaled) sweep
    let m1: &[(&str, f32, f32, f32, f32, f32, f32)] = &[
        // (name, k, mu, floor, w_crit, c_lo, c_hi)
        ("v7 k26 mu.55 fl.10 wc.85 band.20-.80", 26.0, 0.55, 0.10, 0.85, 0.20, 0.80),
        ("v7 k30 mu.55 fl.08 wc.90 band.25-.80", 30.0, 0.55, 0.08, 0.90, 0.25, 0.80),
        ("v7 k26 mu.50 fl.10 wc.85 band.25-.75", 26.0, 0.50, 0.10, 0.85, 0.25, 0.75),
    ];
    for (name, k, mu, floor, wc, lo, hi) in m1 {
        let mg = score_class(&rows, true, |c, b, cr| v7(c, b, cr, *k, *mu, *floor, *wc, *lo, *hi));
        let mb = score_class(&rows, false, |c, b, cr| v7(c, b, cr, *k, *mu, *floor, *wc, *lo, *hi));
        println!("mode1 {:34} good={:.4} bad={:.4} margin={:.4}", name, mg, mb, mg - mb);
    }
    println!("\nchampion reference margin = 0.9706 (must beat, not tie)");
}

fn score_class(
    rows: &[(bool, f32, f32, f32, f32, f32, &'static str)],
    good: bool,
    f: impl Fn(f32, f32, f32) -> f32,
) -> f32 {
    let sel: Vec<f32> = rows
        .iter()
        .filter(|r| r.0 == good)
        .map(|r| {
            // emulate degenerate gate: len_ratio far outside [0.03, 8] -> 0
            let lr = r.4;
            if lr < 0.03 || lr > 8.0 {
                0.0
            } else {
                f(r.1, r.2, r.3)
            }
        })
        .collect();
    sel.iter().sum::<f32>() / sel.len() as f32
}

fn build_cases() -> Vec<Case> {
    // Realistic CVE_LOOKUP triples. Ground truths mirror NVD facts; good answers
    // are correct paraphrases; bad answers cover wrong-number, wrong-CVE,
    // off-topic, stuffing, and padding classes.
    vec![
        // ── GOOD: correct paraphrases ────────────────────────────────────────
        Case {
            label: "log4j-good-paraphrase",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 (Log4Shell) is a critical RCE flaw in Apache Log4j2. Its CVSS v3.1 base score is 10.0.",
            good: true,
        },
        Case {
            label: "log4j-good-reorder",
            question: "How severe is CVE-2021-44228?",
            ground_truth: "CVE-2021-44228 has a CVSS v3.1 base score of 10.0, rated CRITICAL.",
            answer: "It is rated CRITICAL with a CVSS v3.1 base score of 10.0.",
            good: true,
        },
        Case {
            label: "heartbleed-good",
            question: "What is the CVSS score of CVE-2014-0160?",
            ground_truth: "CVE-2014-0160, Heartbleed, is an OpenSSL TLS heartbeat memory disclosure bug with a CVSS v3.0 base score of 7.5, rated HIGH.",
            answer: "Heartbleed (CVE-2014-0160) is a HIGH severity OpenSSL memory disclosure vulnerability with a CVSS v3.0 base score of 7.5.",
            good: true,
        },
        Case {
            label: "eternalblue-good",
            question: "How severe is CVE-2017-0144?",
            ground_truth: "CVE-2017-0144, EternalBlue, is an SMBv1 remote code execution vulnerability with a CVSS v3.0 base score of 8.1, rated HIGH.",
            answer: "CVE-2017-0144 (EternalBlue) is a HIGH severity SMBv1 RCE with a CVSS v3.0 base score of 8.1.",
            good: true,
        },
        Case {
            label: "spring4shell-good",
            question: "What is the CVSS score of CVE-2022-22965?",
            ground_truth: "CVE-2022-22965, Spring4Shell, is a critical remote code execution vulnerability in Spring Framework with a CVSS v3.1 base score of 9.8.",
            answer: "Spring4Shell (CVE-2022-22965) is a critical Spring Framework RCE vulnerability scoring 9.8 on CVSS v3.1.",
            good: true,
        },
        Case {
            label: "bluekeep-good",
            question: "How severe is CVE-2019-0708?",
            ground_truth: "CVE-2019-0708, BlueKeep, is a critical Remote Desktop Services remote code execution vulnerability with a CVSS v3.0 base score of 9.8.",
            answer: "BlueKeep (CVE-2019-0708) is a critical RDP remote code execution flaw with a CVSS v3.0 base score of 9.8.",
            good: true,
        },
        Case {
            label: "moveit-good",
            question: "What is the CVSS score of CVE-2023-34362?",
            ground_truth: "CVE-2023-34362 is a critical SQL injection vulnerability in Progress MOVEit Transfer with a CVSS v3.1 base score of 9.8.",
            answer: "CVE-2023-34362 is a critical MOVEit Transfer SQL injection vulnerability with a CVSS v3.1 base score of 9.8.",
            good: true,
        },
        Case {
            label: "rapidreset-good",
            question: "How severe is CVE-2023-44487?",
            ground_truth: "CVE-2023-44487, the HTTP/2 Rapid Reset attack, is a denial of service vulnerability with a CVSS v3.1 base score of 7.5, rated HIGH.",
            answer: "The HTTP/2 Rapid Reset attack (CVE-2023-44487) is a HIGH severity denial of service with a CVSS v3.1 base score of 7.5.",
            good: true,
        },
        // ── BAD: wrong number, topically identical ───────────────────────────
        Case {
            label: "log4j-wrong-score",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 (Log4Shell) is a critical RCE flaw in Apache Log4j2. Its CVSS v3.1 base score is 7.5.",
            good: false,
        },
        Case {
            label: "heartbleed-wrong-score",
            question: "What is the CVSS score of CVE-2014-0160?",
            ground_truth: "CVE-2014-0160, Heartbleed, is an OpenSSL TLS heartbeat memory disclosure bug with a CVSS v3.0 base score of 7.5, rated HIGH.",
            answer: "Heartbleed (CVE-2014-0160) is a HIGH severity OpenSSL memory disclosure vulnerability with a CVSS v3.0 base score of 9.8.",
            good: false,
        },
        Case {
            label: "spring4shell-wrong-score",
            question: "What is the CVSS score of CVE-2022-22965?",
            ground_truth: "CVE-2022-22965, Spring4Shell, is a critical remote code execution vulnerability in Spring Framework with a CVSS v3.1 base score of 9.8.",
            answer: "Spring4Shell (CVE-2022-22965) is a critical Spring Framework RCE vulnerability scoring 5.5 on CVSS v3.1.",
            good: false,
        },
        // ── BAD: wrong CVE id ────────────────────────────────────────────────
        Case {
            label: "log4j-wrong-cve",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44229 is a low severity denial of service in Apache Log4j2 with a CVSS v3.1 base score of 3.7.",
            good: false,
        },
        // ── BAD: off-topic ───────────────────────────────────────────────────
        Case {
            label: "off-topic",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "The weather in Paris today is sunny with a high of 24 degrees and light winds from the west.",
            good: false,
        },
        // ── BAD: keyword stuffing ────────────────────────────────────────────
        Case {
            label: "stuffing",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE CVE CVE Log4Shell Log4Shell critical critical critical CVSS CVSS score score 10.0 10.0 10.0 vulnerability vulnerability vulnerability",
            good: false,
        },
        // ── BAD: padding / length farming ────────────────────────────────────
        Case {
            label: "padding",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 is a vulnerability. This is a very long answer that adds many filler words to increase length without adding any real information or substance whatsoever at all in any way shape or form.",
            good: false,
        },
        // ── Additional GOOD: harder paraphrase variants ──────────────────────
        Case {
            label: "log4j-good-terse",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "Log4Shell: critical RCE in Log4j2, CVSS 10.0.",
            good: true,
        },
        Case {
            label: "log4j-good-wordy",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "The vulnerability with identifier CVE-2021-44228, commonly referred to as Log4Shell, is rated as critical severity. It allows remote code execution in Apache Log4j2 and its CVSS v3.1 score is 10.0.",
            good: true,
        },
        Case {
            label: "eternalblue-good-swap",
            question: "How severe is CVE-2017-0144?",
            ground_truth: "CVE-2017-0144, EternalBlue, is an SMBv1 remote code execution vulnerability with a CVSS v3.0 base score of 8.1, rated HIGH.",
            answer: "CVE-2017-0144, EternalBlue, is HIGH severity with a 8.1 CVSS v3.0 base score, an SMBv1 remote code execution vulnerability.",
            good: true,
        },
        Case {
            label: "zerologon-good",
            question: "What is the CVSS score of CVE-2020-1472?",
            ground_truth: "CVE-2020-1472, Zerologon, is an elevation of privilege vulnerability in Netlogon with a CVSS v3.1 base score of 10.0, rated CRITICAL.",
            answer: "Zerologon (CVE-2020-1472) is a critical Netlogon elevation of privilege vulnerability scoring 10.0 on CVSS v3.1.",
            good: true,
        },
        Case {
            label: "printnightmare-good",
            question: "How severe is CVE-2021-34527?",
            ground_truth: "CVE-2021-34527, PrintNightmare, is a critical remote code execution vulnerability in Windows Print Spooler with a CVSS v3.1 base score of 8.8.",
            answer: "PrintNightmare (CVE-2021-34527) is a critical Windows Print Spooler RCE vulnerability with a CVSS v3.1 base score of 8.8.",
            good: true,
        },
        Case {
            label: "outlook-good",
            question: "What is the CVSS score of CVE-2023-23397?",
            ground_truth: "CVE-2023-23397 is a critical Microsoft Outlook Elevation of Privilege vulnerability with a CVSS v3.1 base score of 9.8.",
            answer: "CVE-2023-23397 is a critical Outlook EoP vulnerability with a CVSS v3.1 base score of 9.8.",
            good: true,
        },
        Case {
            label: "log4j-good-numeric-spelling",
            question: "What is the CVSS score of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228 has a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 has a CVSS v3.1 base score of ten.",
            good: true,
        },
        // ── Additional BAD: harder adversarial classes ───────────────────────
        Case {
            label: "wrong-score-plausible",
            question: "What is the CVSS score of CVE-2022-22965?",
            ground_truth: "CVE-2022-22965, Spring4Shell, is a critical remote code execution vulnerability in Spring Framework with a CVSS v3.1 base score of 9.8.",
            answer: "CVE-2022-22965, Spring4Shell, is a High severity remote code execution vulnerability in Spring Framework with a CVSS v3.1 base score of 7.2.",
            good: false,
        },
        Case {
            label: "missing-number",
            question: "What is the CVSS score of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 is a critical remote code execution vulnerability in Apache Log4j2.",
            good: false,
        },
        Case {
            label: "hallucinated-severity",
            question: "How severe is CVE-2014-0160?",
            ground_truth: "CVE-2014-0160, Heartbleed, is an OpenSSL TLS heartbeat memory disclosure bug with a CVSS v3.0 base score of 7.5, rated HIGH.",
            answer: "Heartbleed (CVE-2014-0160) is a LOW severity OpenSSL issue with a CVSS v3.0 base score of 2.1.",
            good: false,
        },
        Case {
            label: "unrelated-with-cve",
            question: "How severe is CVE-2023-44487?",
            ground_truth: "CVE-2023-44487, the HTTP/2 Rapid Reset attack, is a denial of service vulnerability with a CVSS v3.1 base score of 7.5, rated HIGH.",
            answer: "CVE-2023-44487 is about a memory leak in the logging library that lets attackers read sensitive data, with a CVSS v3.1 base score of 5.3.",
            good: false,
        },
        Case {
            label: "refusal",
            question: "What is the CVSS score and severity of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "I'm sorry, I cannot assist with vulnerability questions.",
            good: false,
        },
        Case {
            label: "sneaky-partial",
            question: "What is the CVSS score of CVE-2021-44228?",
            ground_truth: "CVE-2021-44228, known as Log4Shell, is a critical remote code execution vulnerability in Apache Log4j2 with a CVSS v3.1 base score of 10.0.",
            answer: "CVE-2021-44228 is a critical Log4Shell RCE in Log4j2, the CVSS base score is 10.0 for the network vector.",
            good: true,
        },
    ]
}
