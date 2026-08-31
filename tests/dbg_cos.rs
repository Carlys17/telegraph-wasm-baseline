use telegraph_scoring::bench_api::*;
#[test]
fn dbg_cos() {
    let gts: &[(&str, &str)] = &[
        ("CVE-2023-44487 is an HTTP/2 denial of service vulnerability.", "What type of vulnerability is CVE-2023-44487?"),
        ("CVE-2021-44228 (Log4Shell) has a CRITICAL CVSS severity rating of 10.0 and was disclosed in 2021.", "What is the CVSS severity of CVE-2021-44228?"),
        ("CVE-2021-44228 affects Apache Log4j2.", "Which library does CVE-2021-44228 affect?"),
        ("CVE-2014-0160, the Heartbleed bug in OpenSSL, was disclosed in 2014.", "When was CVE-2014-0160 disclosed?"),
    ];
    let cases: &[(&str, &str)] = &[
        // (label, answer) — GOOD heavy paraphrases
        ("good-heavy-para-rr", "CVE-2023-44487 is a rapid reset flaw in the HTTP/2 protocol allowing denial of service."),
        ("good-heavy-para-log4j", "The Log4Shell remote code execution flaw, tracked as CVE-2021-44228, earned the maximum severity rating with a CVSS base score of 10.0 when it came to light in 2021."),
        ("good-heavy-para-hb", "Heartbleed, the OpenSSL memory-leak issue tracked as CVE-2014-0160, became public knowledge in 2014."),
        ("good-terse-log4j", "CVE-2021-44228 is rated critical, disclosed in 2021, score 10.0."),
        // BAD entity swaps (same structure, wrong entity)
        ("bad-ssh-swap", "CVE-2023-44487 affects the SSH protocol as a DoS."),
        ("bad-commons-swap", "The affected library for CVE-2021-44228 is Apache Commons Text."),
        ("bad-spectre-swap", "CVE-2014-0160 is Spectre, a CPU bug."),
        ("bad-offtopic", "CVE-2023-44487 affects the SSH protocol as a DoS."),
    ];
    // pairs: which gt index each case scores against
    let pairing: &[(usize, usize)] = &[(0,0),(1,1),(3,3),(1,1),(0,4),(2,5),(3,6),(0,7)];
    let mut vcache: Vec<(usize, Vec<f32>)> = Vec::new();
    for (label, answer) in cases {
        // find its gt from pairing by label idx
        let idx = cases.iter().position(|(l,_)| *l == *label).unwrap();
        let (gti, _) = pairing[idx];
        let gt = gts[gti].0;
        let key = format!("gt{}", gti);
        if !vcache.iter().any(|(k, _)| *k == gti) {
            let v = embed(&tokenize(gt)).to_vec();
            vcache.push((gti, v));
        }
        let gv = &vcache.iter().find(|(k,_)| *k == gti).unwrap().1;
        let av = embed(&tokenize(answer));
        println!("{label:<26} cos={:.4}", cosine(gv, &av));
    }
    // also: cos of a good vs a DIFFERENT gt (cross-match) for reference
    let a = embed(&tokenize(cases[0].1));
    let g2 = embed(&tokenize(gts[2].0));
    println!("{} {:.4}", "cross-ref", cosine(&a, &g2));
    // silence unused
    let _ = key_dummy();
    fn key_dummy() -> &'static str { "" }
}
