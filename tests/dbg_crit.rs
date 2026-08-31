
//! Why is vector-good crit 0.857 not 1.0? GT figures: 9.8 (anchored cvss), 3.0 (version), 2019,0708 (cve). Answer: "BlueKeep scores 9.8 CVSS and can be exploited over the network."
use telegraph_scoring::bench_api::*;
#[test]
fn dbg_crit_breakdown() {
    let gt = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with a network attack vector.";
    let good = "BlueKeep scores 9.8 CVSS and can be exploited over the network.";
    let bad  = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with an adjacent attack vector.";
    println!("good crit={} bad crit={}", critical_token_match(gt, good), critical_token_match(gt, bad));
    // Try variants stripping parts of GT to see which figure loses weight
    let variants: &[&str] = &[
        "CVE-2019-0708 has a CVSS base score of 9.8 with a network attack vector.",
        "CVE-2019-0708 has a CVSS v3.0 base score of 9.8.",
        "CVE-2019-0708 has a base score of 9.8 with a network attack vector.",
    ];
    for (i, v) in variants.iter().enumerate() {
        println!("variant {i} good crit={:.4}", critical_token_match(v, good));
    }
}
