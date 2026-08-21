//! Native separation benchmark for the v3 composite (no embeddings — uses
//! representative cosine/lexical operating-point values per answer class to
//! measure average margin the same way Stage 2 does: mean(good) - mean(bad).
//!
//! Run: rustc -O --edition 2021 tools/sep_bench.rs -o /tmp/sep && /tmp/sep

fn clamp01(v: f32) -> f32 { if v < 0.0 {0.0} else if v > 1.0 {1.0} else {v} }
fn sharpen(raw: f32, k: f32, mu: f32) -> f32 { clamp01(1.0/(1.0+(-k*(raw-mu)).exp())) }

const K: f32 = 14.0;
const MU: f32 = 0.55;
const FLOOR: f32 = 0.15;

// v3: lexical-gated correctness then steep sharpen
fn v3(correctness: f32, lexical: f32) -> f32 {
    let evidence = correctness * (FLOOR + (1.0 - FLOOR) * lexical);
    sharpen(evidence, K, MU)
}

// v2 (rejected): linear blend * smooth penalty (penalty ~0.85 typical clean)
fn v2(relevance: f32, correctness: f32, lexical: f32, len_q: f32, penalty: f32) -> f32 {
    clamp01(0.25*relevance + 0.50*correctness + 0.15*lexical + 0.10*len_q) * penalty
}

fn main() {
    // (relevance, correctness, lexical=0.5*bm25+0.5*crit, len_q, penalty, class)
    // class: true = good answer, false = bad. Operating points reflect MiniLM
    // anisotropy: good near-duplicate c~0.85-0.95 l~0.85-1.0; wrong-number
    // c~0.75-0.85 (topically identical) l~0.2-0.35 (crit tokens miss);
    // off-topic c~0.25-0.45 l~0.05-0.2.
    let cases: &[(f32,f32,f32,f32,f32,bool)] = &[
        // GOOD (correct, complete, right numbers)
        (0.80,0.92,0.95,1.0,1.0,true),
        (0.78,0.88,0.90,1.0,1.0,true),
        (0.82,0.90,1.00,1.0,1.0,true),
        (0.75,0.85,0.85,0.9,1.0,true),
        (0.79,0.93,0.92,1.0,1.0,true),
        (0.77,0.87,0.88,1.0,1.0,true),
        // BAD: topically identical but WRONG numbers (the hard case)
        (0.74,0.82,0.25,1.0,1.0,false),
        (0.72,0.80,0.30,1.0,1.0,false),
        (0.76,0.84,0.20,1.0,1.0,false),
        // BAD: off-topic
        (0.35,0.35,0.10,0.8,1.0,false),
        (0.28,0.30,0.05,0.7,1.0,false),
        // BAD: keyword-stuffed (high lexical, gamed) — v2 penalty ~0.3, v3 gate=0
        (0.40,0.55,0.80,1.0,0.30,false),
        // BAD: padded filler
        (0.45,0.50,0.20,0.5,0.85,false),
    ];

    let mut g3=Vec::new(); let mut b3=Vec::new();
    let mut g2=Vec::new(); let mut b2=Vec::new();
    for &(r,c,l,lq,p,good) in cases {
        // v3 gates stuffed/padded to 0 (degenerate); emulate: penalty<0.5 => gated
        let s3 = if p < 0.5 { 0.0 } else { v3(c,l) };
        let s2 = v2(r,c,l,lq,p);
        if good { g3.push(s3); g2.push(s2); } else { b3.push(s3); b2.push(s2); }
    }
    let mean=|v:&Vec<f32>| v.iter().sum::<f32>()/v.len() as f32;
    println!("v3: mean(good)={:.4} mean(bad)={:.4} margin={:.4}", mean(&g3),mean(&b3),mean(&g3)-mean(&b3));
    println!("v2: mean(good)={:.4} mean(bad)={:.4} margin={:.4}", mean(&g2),mean(&b2),mean(&g2)-mean(&b2));
    println!("champion reference margin = 0.8081");
    // per-case dump
    println!("\n-- v3 per case (class,score) --");
    for &(_r,c,l,_lq,p,good) in cases {
        let s3 = if p < 0.5 { 0.0 } else { v3(c,l) };
        println!("  {} c={:.2} l={:.2} -> {:.3}", if good {"GOOD"} else {"BAD "}, c, l, s3);
    }
}
