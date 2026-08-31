
//! Debug v17 crush decisions on battery cases.
use telegraph_scoring::bench_api::*;
#[test]
fn dbg_v17_leaks() {
    // vector-string-mismatch GOOD — got 0.023 in wasm battery, expected ~0.99
    let gt = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with a network attack vector.";
    let good = "BlueKeep scores 9.8 CVSS and can be exploited over the network.";
    let bad  = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with an adjacent attack vector.";
    for (label, a) in [("good", good), ("bad", bad)] {
        let crit = critical_token_match(gt, a);
        let cm = claim_mismatch(gt, a);
        let b = bm25(gt, a);
        let lr = answer_len_ratio(gt, a);
        println!("{label:<5} mismatch={cm} crit={crit:.4} bm25={b:.4} lenr={lr:.2}");
    }
    // log4j-affects: entity swap bad rides at 0.6873
    let gt2 = "CVE-2021-44228 affects Apache Log4j2.";
    let good2 = "The affected component is Apache's Log4j2 logging library.";
    let bad2 = "CVE-2021-44228 affects Apache Commons Text.";
    for (label, a) in [("good", good2), ("bad", bad2)] {
        let crit = critical_token_match(gt2, a);
        let cm = claim_mismatch(gt2, a);
        let b = bm25(gt2, a);
        println!("affects {label:<5} mismatch={cm} crit={crit:.4} bm25={b:.4}");
    }
}
