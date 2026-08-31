use telegraph_scoring::bench_api::*;

fn main() {
    let gt = "CVE-2021-44228 (Log4Shell) has a CRITICAL CVSS severity rating of 10.0 and was disclosed in 2021.";
    let q = "What is the CVSS severity of CVE-2021-44228?";
    let cases = [
        ("p1-complete-paraphrase", "The remote code execution vulnerability Log4Shell, identified as CVE-2021-44228, is rated critical and carries the maximum CVSS score of 10.0. It was made public in 2021."),
        ("p3-complete", "Log4Shell is a critical-severity flaw. Its CVSS base score is 10.0 and it came to light in 2021 under the identifier CVE-2021-44228."),
        ("exact", gt),
        ("only-cve", "CVE-2021-44228 is a vulnerability."),
        ("p8-affected", "Apache Log4j from 2.0-beta9 up to and including 2.14.1 is impacted by CVE-2021-44228."),
    ];
    println!("{:<24} {:>6} {:>6} {:>6} {:>6} {:>6}", "case", "rel", "corr", "bm25", "crit", "len");
    for (label, a) in cases {
        let (qq, gg) = if label.starts_with("p8") {
            ("Which versions does CVE-2021-44228 affect?", "CVE-2021-44228 affects Apache Log4j versions 2.0-beta9 through 2.14.1.")
        } else { (q, gt) };
        let (rel, corr, b, crit, lr) = signals(qq, gg, a);
        println!("{:<24} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.2}", label, rel, corr, b, crit, lr);
    }
}
