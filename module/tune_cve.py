#!/usr/bin/env python3
"""Sweep champion-scorer constants against the CVE benchmark via the Node harness.
Patches lib.rs in place, rebuilds wasm, runs run_wasm.js, captures candidate_margin.
"""
import itertools, json, re, subprocess, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "src", "lib.rs")
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")
BENCH = os.path.join(ROOT, "cve_benchmark.json")
RUNNER = os.path.join(ROOT, "run_wasm.js")

BASE_SRC = open(LIB).read()

def patch_consts(src, values):
    for name, val in values.items():
        src, n = re.subn(rf"const {name}: f32 = [0-9.]+;", f"const {name}: f32 = {val};", src)
        if n != 1:
            raise SystemExit(f"could not patch {name}")
    return src

def build():
    r = subprocess.run(
        ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAIL", r.stderr[-500:]); return False
    return True

def run_margin():
    r = subprocess.run(["node", RUNNER, WASM, BENCH], capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"candidate_margin ([0-9.]+)", out)
    wins = re.search(r"wins (\d+)/(\d+)", out)
    worst = re.search(r"worst_self_match ([0-9.]+)", out)
    if not m:
        return None, None, None, out
    return float(m.group(1)), (int(wins.group(1)), int(wins.group(2))) if wins else None, float(worst.group(1)) if worst else None, out

def main():
    # Grid: focus on numeric penalties + sharpening + recall balance (CVE is numeric-heavy)
    grid = {
        "M_NUM_MISS_BASE": [0.62, 0.45, 0.30],
        "M_NUM_WRONG":     [0.45, 0.25, 0.12],
        "SHARPEN":         [0.82, 0.90],
        "F_BETA2":         [0.6],
        "R_KEY_BASE":      [0.6],
        "M_CONTRA":        [0.3, 0.15],
        "B_AGREE":         [0.35],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"sweeping {len(combos)} combos", flush=True)
    results = []
    for i, combo in enumerate(combos):
        values = dict(zip(keys, combo))
        open(LIB, "w").write(patch_consts(BASE_SRC, values))
        if not build():
            continue
        margin, wins, worst, _ = run_margin()
        if margin is None:
            continue
        results.append((margin, wins, worst, values))
        print(f"[{i+1}/{len(combos)}] margin={margin:.4f} wins={wins} worst_self={worst:.3f} {values}", flush=True)
    # restore base
    open(LIB, "w").write(BASE_SRC)
    results.sort(key=lambda x: -x[0])
    print("\n=== TOP 10 ===")
    for margin, wins, worst, values in results[:10]:
        print(f"margin={margin:.4f} wins={wins} worst_self={worst:.3f} {values}")
    json.dump([{"margin":m,"wins":w,"worst_self":ws,"values":v} for m,w,ws,v in results],
              open(os.path.join(ROOT,"sweep_results.json"),"w"), indent=1)

if __name__ == "__main__":
    main()
