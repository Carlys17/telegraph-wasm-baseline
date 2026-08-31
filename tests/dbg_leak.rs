
//! Dump signals for leaking good answers.
use telegraph_scoring::bench_api::*;
#[test]
fn dbg_leak_signals() {
    let cases: &[(&str, &str, &str)] = &[
        ("rr-score-good", "CVE-2023-44487, the HTTP/2 Rapid Reset attack, is a denial of service vulnerability with a CVSS v3.1 base score of 7.5, rated HIGH.",
         "The HTTP/2 Rapid Reset attack (CVE-2023-44487) is a HIGH severity denial of service with a CVSS v3.1 base score of 7.5."),
        ("outlook-good", "CVE-2023-23397 is a critical Microsoft Outlook Elevation of Privilege vulnerability with a CVSS v3.1 base score of 9.8.",
         "CVE-2023-23397 is a critical Outlook EoP vulnerability with a CVSS v3.1 base score of 9.8."),
        ("vector-good", "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with a network attack vector.",
         "BlueKeep scores 9.8 CVSS and can be exploited over the network."),
        ("rr-type-good", "CVE-2023-44487 is a denial of service vulnerability in HTTP/2.",
         "It is a denial of service issue in the HTTP/2 protocol."),
        ("prod-swap-good", "CVE-2021-44228 affects Apache Log4j2.",
         "The affected component is Apache's Log4j2 logging library."),
    ];
    for (label, gt, a) in cases {
        let crit = critical_token_match(gt, a);
        let b = bm25(gt, a);
        let lr = answer_len_ratio(gt, a);
        let gv = embed(&tokenize(gt)); let av = embed(&tokenize(a));
        let c = cosine(&gv, &av);
        let l = 0.2*b + 0.8*crit;
        let c_norm = ((c - 0.55)/0.30).clamp(0.0,1.0);
        let ev = l * (0.5 + 0.5*c_norm);
        println!("{label:<16} crit={crit:.4} bm25={b:.4} cos={c:.4} lenr={lr:.2} l={l:.3} ev={ev:.3}");
    }
}
