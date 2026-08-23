#!/usr/bin/env node
// Loads a Telegraph scoring .wasm exactly the way the node does:
// alloc -> write strings -> rank_answer. No host imports.
// usage: node run_wasm.js <module.wasm> <benchmark.json> [probe:"q|gt|a"]

const fs = require('fs');

async function loadMod(path) {
  const bytes = fs.readFileSync(path);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const mem = instance.exports.memory;
  const alloc = instance.exports.alloc;
  const rank = instance.exports.rank_answer;
  const dealloc = instance.exports.dealloc;
  if (!alloc || !rank || !dealloc || !mem) {
    throw new Error('missing export: alloc, dealloc, rank_answer and linear memory are all required');
  }
  if (instance.exports.TELEGRAPH_INTENT) {
    const off = instance.exports.TELEGRAPH_INTENT.value || instance.exports.TELEGRAPH_INTENT;
    const len = 32;
    const buf = new Uint8Array(mem.buffer, off, len);
    const intent = String.fromCharCode(...buf).replace(/\s+$/, '');
    console.error(`[intent] ${intent}`);
  }
  return { instance, mem, alloc, rank, dealloc };
}

function put(m, s) {
  const enc = Buffer.from(s, 'utf8');
  const p = m.alloc(enc.length);
  // copy into wasm memory
  new Uint8Array(m.mem.buffer, p, enc.length).set(enc);
  return [p, enc.length];
}

function score(m, q, gt, ans) {
  const [qp, ql] = put(m, q);
  const [gp, gl] = put(m, gt);
  const [ap, al] = put(m, ans);
  const v = m.rank(qp, ql, gp, gl, ap, al);
  m.dealloc(qp, ql); m.dealloc(gp, gl); m.dealloc(ap, al);
  return v;
}

function stddev(v) {
  if (v.length < 2) return 0;
  const m = v.reduce((a,b)=>a+b,0)/v.length;
  return Math.sqrt(v.reduce((a,b)=>a+(b-m)*(b-m),0)/(v.length-1));
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length < 2) {
    console.error('usage: node run_wasm.js <module.wasm> <benchmark.json> [probe]');
    process.exit(2);
  }
  const [wasmPath, benchPath] = argv;

  if (process.env.PROBE) {
    const parts = process.env.PROBE.split('|');
    const m = await loadMod(wasmPath);
    const s = score(m, parts[0], parts[1], parts[2]);
    console.log(s.toFixed(6));
    return;
  }

  const bench = JSON.parse(fs.readFileSync(benchPath, 'utf8'));
  const cases = bench.cases || bench.rows || [];
  if (!cases.length) { console.error('no cases'); process.exit(2); }

  const m = await loadMod(wasmPath);

  // Stage 1 gates (mirror champion harness)
  let failed = [];
  const add = (name, ok, detail) => { console.log(`  ${ok?'[ok]  ':'[FAIL]'} ${name}  ${detail}`); if(!ok) failed.push(name); };
  const c0 = cases[0];
  let s0 = score(m, c0.question||c0.q, c0.ground_truth||c0.gt, '');
  add('empty answer scores exactly 0', s0 === 0, `score=${s0.toFixed(4)}`);
  s0 = score(m, c0.question||c0.q, c0.ground_truth||c0.gt, ' \t\n \r ');
  add('whitespace-only answer scores exactly 0', s0 === 0, `score=${s0.toFixed(4)}`);

  let worstSelf = 1.0, worstGap = 1.0, gapCase = '';
  for (let i=0;i<cases.length;i++){
    const c = cases[i];
    const q=c.question||c.q, gt=c.ground_truth||c.gt;
    const self = score(m,q,gt,gt);
    const cross = score(m,q,gt,cases[(i+1)%cases.length].ground_truth||cases[(i+1)%cases.length].gt);
    if (self < worstSelf) worstSelf = self;
    if (self-cross < worstGap){ worstGap=self-cross; gapCase=c.id||c.key||('case'+i); }
  }
  add('perfect answer scores >= 0.75 everywhere', worstSelf >= 0.75, `worst_self_match=${worstSelf.toFixed(4)}`);
  add('self-match beats unrelated cross-match', worstGap > 0, `narrowest gap ${worstGap.toFixed(4)} (${gapCase})`);

  const long = 'lorem ipsum dolor sit amet consectetur adipiscing '.repeat(1600);
  let sl = score(m, c0.question||c0.q, c0.ground_truth||c0.gt, long);
  add('long answer does not trap', sl >= 0 && sl <= 1, `score=${sl.toFixed(4)}`);
  const weird = '🚀🌙 登月成功了 مرحبا بالعالم Привет мир ✅ \x00\xff\xfe binary';
  let sw = score(m, c0.question||c0.q, c0.ground_truth||c0.gt, weird);
  add('weird answer does not trap', sw >= 0 && sw <= 1, `score=${sw.toFixed(4)}`);

  // Stage 2 separation
  console.log('\n-- separation --');
  let sumGood=0,sumBad=0,sumMargin=0, ties=0, wins=0;
  const per = {};
  const all = [];
  for (const c of cases){
    const q=c.question||c.q, gt=c.ground_truth||c.gt;
    const good = score(m,q,gt,c.good);
    const bad = score(m,q,gt,c.bad);
    const id = c.id||c.key||('case'+per);
    per[id]=[good,bad];
    if (good>bad) wins++; else if (good===bad) ties++;
    sumGood+=good; sumBad+=bad; sumMargin+=good-bad; all.push(good,bad);
  }
  const n = cases.length;
  const margin = sumMargin/n, meanGood=sumGood/n, meanBad=sumBad/n;
  console.log(`  comparable_cases ${n} | wins ${wins}/${n} | ties ${ties}`);
  console.log(`  candidate_margin ${margin.toFixed(4)} | mean good ${meanGood.toFixed(4)} bad ${meanBad.toFixed(4)}`);
  console.log(`  worst_self_match ${worstSelf.toFixed(4)} | score_stddev ${stddev(all).toFixed(4)}`);

  // per-case detail
  console.log('\n-- per case --');
  for (const c of cases){
    const id = c.id||c.key;
    const [g,b] = per[id];
    console.log(`  ${String(id).padEnd(28)} good=${g.toFixed(4)} bad=${b.toFixed(4)} margin=${(g-b).toFixed(4)}`);
  }
}

main().catch(e=>{ console.error('FATAL', e); process.exit(1); });
