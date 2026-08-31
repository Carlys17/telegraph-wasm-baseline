//! Debug critical_token_match / claim_mismatch natively.
use telegraph_scoring::bench_api::*;
#[test]
fn dbg_partial() {
    let gt = "CVE-2021-44228 (Log4Shell) has a CRITICAL CVSS severity rating of 10.0 and was disclosed in 2021.";
    for (label, a) in [
        ("partial-crit-year", "CVE-2021-44228 has a CVSS score of 10.0 and was disclosed in 2021."),
        ("partial-crit", "CVE-2021-44228 is rated critical and disclosed in 2021."),
        ("complete", "CVE-2021-44228 is rated critical, disclosed in 2021, score 10.0."),
    ] {
        let crit = critical_token_match(gt, a);
        let cm = claim_mismatch(gt, a);
        let b = bm25(gt, a);
        println!("{label:<20} crit={crit:.4} mismatch={cm} bm25={b:.4}");
    }
}
