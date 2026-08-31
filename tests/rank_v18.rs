use telegraph_scoring::bench_api::score_answer;

fn score(gt: &str, answer: &str) -> f32 {
    score_answer(gt, gt, answer)
}

#[test]
fn v18_regressions() {
    let gt_vec = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with a network attack vector.";
    let good_vec = "BlueKeep scores 9.8 CVSS and can be exploited over the network.";
    let bad_vec = "CVE-2019-0708 has a CVSS v3.0 base score of 9.8 with an adjacent attack vector.";
    let gt_entity = "CVE-2021-44228 affects Apache Log4j2.";
    let good_entity = "The affected component is Apache's Log4j2 logging library.";
    let bad_entity = "CVE-2021-44228 affects Apache Commons Text.";
    let good = score(gt_vec, good_vec);
    let bad = score(gt_vec, bad_vec);
    let egood = score(gt_entity, good_entity);
    let ebad = score(gt_entity, bad_entity);
    println!("vector good={good:.6} bad={bad:.6}; entity good={egood:.6} bad={ebad:.6}");
    assert!(good > 0.9, "vector paraphrase regressed: {good}");
    assert!(bad < 0.1, "vector mismatch not crushed: {bad}");
    assert!(egood > 0.5, "entity good rejected: {egood}");
    assert!(ebad < 0.2, "entity swap not rejected: {ebad}");
}

