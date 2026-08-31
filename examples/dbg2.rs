use telegraph_scoring::bench_api::*;
fn main() {
    let gt = "CVE-2021-44228 (Log4Shell) has a CRITICAL CVSS severity rating of 10.0 and was disclosed in 2021.";
    let q = "What is the CVSS severity of CVE-2021-44228?";
    for (label, a) in [
        ("partial-crit-year", "CVE-2021-44228 has a CVSS score of 10.0 and was disclosed in 2021."),
        ("partial-crit", "CVE-2021-44228 is rated critical and disclosed in 2021."),
        ("complete", "CVE-2021-44228 is rated critical, disclosed in 2021, score 10.0."),
        ("only-cve", "CVE-2021-44228 is a vulnerability."),
    ] {
        let (rel, corr, b, crit, lr) = signals(q, gt, a);
        println!("{label:<20} rel={rel:.3} corr={corr:.3} bm25={b:.3} crit={crit:.3} lr={lr:.2} claim_mismatch={}", claim_mismatch(gt, a));
    }
}
