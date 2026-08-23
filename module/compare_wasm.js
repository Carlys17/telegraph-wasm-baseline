#!/usr/bin/env node
// Direct comparison: given two wasm files + the same (q, good, bad) triple, print both scores.
// usage: node compare_wasm.js <wasm1> <wasm2> <pairs.json>
const fs = require('fs');

function loadWasm(path) {
  const bytes = fs.readFileSync(path);
  const mod = new WebAssembly.Module(bytes);
  const inst = new WebAssembly.Instance(mod, {});
  const e = inst.exports;
  const mem = e.memory;
  const alloc = e.alloc, dealloc = e.dealloc, rank = e.rank_answer;
  function put(s) {
    const enc = Buffer.from(s, 'utf8');
    const p = alloc(enc.length);
    new Uint8Array(mem.buffer, p, enc.length).set(enc);
    return [p, enc.length];
  }
  function score(q, gt, ans) {
    const [qp, ql] = put(q);
    const [gp, gl] = put(gt);
    const [ap, al] = put(ans);
    const v = rank(qp, ql, gp, gl, ap, al);
    dealloc(qp, ql); dealloc(gp, gl); dealloc(ap, al);
    return v;
  }
  return { score };
}

const [,, w1path, w2path, pairsPath] = process.argv;
const pairs = JSON.parse(fs.readFileSync(pairsPath));
const w1 = loadWasm(w1path);
const w2 = loadWasm(w2path);

for (const c of pairs) {
  const g1 = w1.score(c.q, c.gt, c.good);
  const b1 = w1.score(c.q, c.gt, c.bad);
  const g2 = w2.score(c.q, c.gt, c.good);
  const b2 = w2.score(c.q, c.gt, c.bad);
  console.log(`${c.id}`);
  console.log(`  ${w1path.match(/([^/]+)\.wasm$/)[1]}: good=${g1.toFixed(4)} bad=${b1.toFixed(4)} margin=${(g1-b1).toFixed(4)}`);
  console.log(`  ${w2path.match(/([^/]+)\.wasm$/)[1]}: good=${g2.toFixed(4)} bad=${b2.toFixed(4)} margin=${(g2-b2).toFixed(4)}`);
}
