use telegraph_scoring::bench_api::*;
#[test]
fn dbg_log4j_affects() {
    let gt = "CVE-2021-44228 affects Apache Log4j2.";
    let good = "The affected library for CVE-2021-44228 is Apache Log4j2.";
    let bad = "The affected library for CVE-2021-44228 is Apache Commons Text.";
    for (label, a) in [("good", good), ("bad", bad)] {
        let gv = embed(&tokenize(gt)); let av = embed(&tokenize(a));
        let c = cosine(&gv, &av);
        let b = bm25(gt, a);
        let cm = claim_mismatch(gt, a);
        let crit = critical_token_match(gt, a);
        println!("{label}: cos={c:.4} bm25={b:.4} mismatch={cm} crit={crit:.4}");
    }
}
