#!/usr/bin/env python3
"""Sweep SHARPEN on top of the proven v10b contrast (STEP_T=0, POST_ITERS=3).
v12's STEP_T=0.2 collapse regressed separation; this restores v10b and finds
the best SHARPEN strictly above v10b to beat the 0.7253 incumbent."""
import json, re, subprocess, os, sys, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "src", "lib.rs")
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "telegraph_scorer.wasm")
BENCH = os.path.join(ROOT, "cve_benchmark.json")
RUNNER = os.path.join(ROOT, "run_wasm.js")
BACKUP = LIB + ".bak"

shutil.copy(LIB, BACKUP)
BASE_SRC = open(LIB).read()

def set_const(name, val):
    global BASE_SRC
    BASE_SRC, n = re.subn(rf"const {name}: f32 = [0-9.eE+-]+;", f"const {name}: f32 = {val};", BASE_SRC)
    if n != 1:
        raise SystemExit(f"could not patch {name}")

def build():
    r = subprocess.run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-400:]

def measure():
    r = subprocess.run(["node", RUNNER, WASM, BENCH], capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"candidate_margin ([0-9.]+)", out)
    wins = re.search(r"wins (\d+)/(\d+)", out)
    worst = re.search(r"worst_self_match ([0-9.]+)", out)
    return (float(m.group(1)) if m else None,
            (int(wins.group(1)), int(wins.group(2))) if wins else (0,0),
            float(worst.group(1)) if worst else None, out)

results = []
for sharpen in [0.82, 0.84, 0.86, 0.88, 0.90]:
    # restore base then set STEP_T=0 (v10b contrast) + this SHARPEN
    src = open(BACKUP).read()
    BASE_SRC = src
    set_const("STEP_T", 0.0)
    set_const("STEP_B", 0.0)
    set_const("SHARPEN", sharpen)
    open(LIB, "w").write(BASE_SRC)
    ok, err = build()
    if not ok:
        print(f"BUILD FAIL sharpen={sharpen}: {err}"); continue
    margin, wins, worst, out = measure()
    print(f"SHARPEN={sharpen}: margin={margin} wins={wins} worst_self={worst}", flush=True)
    if margin and wins[0]==wins[1] and worst and worst>=0.75:
        results.append((margin, sharpen))

results.sort(key=lambda x: -x[0])
print("\n=== ranked (passing gates) ===")
for margin, sharpen in results:
    print(f"  SHARPEN={sharpen} margin={margin:.4f}")

# pick best passing config, write it to lib.rs, rebuild
if results:
    best_sh, = results[0][1],
    src = open(BACKUP).read(); BASE_SRC = src
    set_const("STEP_T", 0.0); set_const("STEP_B", 0.0); set_const("SHARPEN", best_sh)
    open(LIB, "w").write(BASE_SRC)
    build()
    m, w, worst, _ = measure()
    print(f"\nFINAL SHARPEN={best_sh}: margin={m} wins={w} worst_self={worst}")
    print(f"WASM: {WASM}")
else:
    print("NO PASSING CONFIG — restored v10b baseline (SHARPEN=0.82, STEP_T=0)")
    src = open(BACKUP).read(); BASE_SRC = src
    set_const("STEP_T", 0.0); set_const("STEP_B", 0.0); set_const("SHARPEN", 0.82)
    open(LIB, "w").write(BASE_SRC); build()
