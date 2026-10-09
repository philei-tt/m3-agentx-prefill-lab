#!/usr/bin/env node
// Where do fused attention and KV prefetch save time, and does it reach the pipeline bottleneck?
//   A) per-layer op breakdown for a batch of N requests x 1.6k new tokens at 140k context (sparse MSA layer and dense
//      layer), per-request vs fused attention, prefetch off/on, chunk 128 vs 1024 (request padding)
//   B) full replay at the peak concurrency (infinite cache): which stages are busiest, dense vs MoE, per variant
// Usage: JOB=<slurm job> ./on_node.sh node tools/attn_diag.js
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs'), path = require('path');
const SIM = require('../sim_core.js');

if (!isMainThread) {
  const { loadAll } = require('../run.js');
  const { TR, cal } = loadAll(workerData.data);
  const r = SIM.simulate(TR, cal, workerData.cfg);
  const plan = SIM.makePlan(workerData.cfg, cal);
  parentPort.postMessage({ useful: r.usefulTps, p90: r.ttftP90, util: r.stageUtil, counts: plan.counts, req: r.avgSegsPerChunk });
  return;
}

const cal = SIM.calibrate(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'calib_data.json'))));
const study = JSON.parse(fs.readFileSync(require('../lib/paths.js').STUDY));
const { withFeatures } = require('../study.js');

// ---- A) op breakdown
function breakdown(label, cfgBase) {
  console.log(`\n== A) ${label}`);
  for (const C of [128, 1024]) for (const N of [1, 4, 10]) {
    const n = Math.ceil(1600 / C) * C, T = N * n;
    const segs = Array.from({ length: N }, () => ({ n, na: 1600, k: 140000, cap: 141600 }));
    const row = [];
    for (const [attn, prefetchKV] of [['seq', false], ['fused', false], ['seq', true], ['fused', true]]) {
      const plan = SIM.makePlan(Object.assign({}, cfgBase, { chunk: C, attn, prefetchKV, batch: true, budget: 65536 }), cal);
      row.push({ attn, prefetchKV, moe: plan.layer('moe', T, segs), dense: plan.layer('dense', T, segs) });
    }
    // op-level parts of one MSA layer's attention (per-request schedule)
    const plan = SIM.makePlan(Object.assign({}, cfgBase, { chunk: C, batch: true, budget: 65536 }), cal);
    const [sp, tp] = cfgBase.mesh;
    const c = { sp, tp, P: sp * tp, T, Tr: N * 1600, idxB: cfgBase.idxBf16 === false ? 1.0625 : 2, imb: SIM.DEFAULTS.expertImb, bounded: true,
      ringSp: sp === 4 && tp === 4, ringTp: sp === 4 && tp === 4 };
    const op = (o, k) => segs.reduce((a, s) => a + SIM.roofSeg(o, c, s) * 1e3 / plan.eff[k][o], 0);
    const gat = op('ag_kv', 'moe') + op('ag_idx', 'moe'), att = op('indexer', 'moe') + op('sparse', 'moe');
    const rc = op('ring_c', 'dense'), rs = op('ring_scan', 'dense');
    const f = (x) => x.toFixed(1).padStart(6);
    console.log(`  C=${String(C).padStart(4)} N=${String(N).padStart(2)} (T=${String(T).padStart(5)}): MoE layer seq ${f(row[0].moe)} fused ${f(row[1].moe)} (${((1 - row[1].moe / row[0].moe) * 100).toFixed(1)}%) | seq+pf ${f(row[2].moe)} (${((1 - row[2].moe / row[0].moe) * 100).toFixed(1)}%)`
      + ` || dense layer seq ${f(row[0].dense)} fused ${f(row[1].dense)} (${((1 - row[1].dense / row[0].dense) * 100).toFixed(1)}%) | seq+pf ${f(row[2].dense)} (${((1 - row[2].dense / row[0].dense) * 100).toFixed(1)}%)`
      + ` || MSA KV/index gather ${f(gat)} vs indexer+sparse ${f(att)} ms; dense ring compute ${f(rc)} vs gather ${f(rs)} ms`);
  }
}
const best = (key) => { const R = study.scenarios[key]; return Object.assign(withFeatures(R.base, R.bestKeys.filter((k) => !['reqPad', 'fused'].includes(k))), R.grid[0].extra, { reqPad: 'chunk', cache: 'inf', hostTier: false, laneArena: false }); };
const topo = (key) => { const g = study.scenarios[key].grid[0].extra; return `${g.stages}x[${g.mesh}]`; };
breakdown(`8 gx today (${topo('g8_k0')}, today's kernels)`, best('g8_k0'));
breakdown(`8 gx roofline (${topo('g8_k1')}, roofline kernels)`, best('g8_k1'));

// ---- B) bottleneck at the peak
// peak concurrency of chunk C=128, 64k, per-request attention (results/layout_ab_inf_peak.json)
const PEAK = {};
for (const [key, rows] of Object.entries(JSON.parse(fs.readFileSync(path.join(require('../lib/paths.js').RESULTS, 'layout_ab_inf_peak.json'))).scenarios)) {
  const r = rows.find((x) => x.mode === 'base' && x.reqPad === 'chunk' && x.attn === 'seq' && x.chunk === 128 && x.budget === 65536);
  if (r && r.at) PEAK[key] = r.at.conc;
}
const jobs = [];
for (const [key, conc] of Object.entries(PEAK)) for (const [attn, prefetchKV] of [['seq', false], ['fused', false], ['seq', true], ['fused', true]])
  jobs.push({ key, attn, prefetchKV, cfg: Object.assign(best(key), { chunk: 128, batch: true, budget: 65536, attn, prefetchKV, concurrency: conc }) });
Promise.all(jobs.map((j) => new Promise((res) => {
  const w = new Worker(__filename, { workerData: { data: require('../lib/paths.js').DATA, cfg: j.cfg } });
  w.on('message', (m) => { res(Object.assign({}, j, m)); w.terminate(); });
}))).then((rs) => {
  console.log('\n== B) replay at the peak concurrency, chunk 128, 64k budget, infinite cache: stage utilisation');
  for (const r of rs) {
    let start = 0; const dense = [], moe = [];
    // a stage holding any of the 3 dense layers (with the auto split it may also hold MoE layers)
    r.counts.forEach((n, s) => { (start < 3 ? dense : moe).push(r.util[s]); start += n; });
    const mx = (a) => (a.length ? (100 * Math.max(...a)).toFixed(0) : '-');
    console.log(`  ${r.key} ${r.attn.padEnd(5)} prefetch ${String(r.prefetchKV).padEnd(5)}: useful ${(r.useful / 1e3).toFixed(1)}k, ${r.req.toFixed(1)} req/chunk | busiest stage with a dense layer ${mx(dense)}% (${dense.length} stages) vs busiest MoE-only stage ${mx(moe)}% | layers/stage ${r.counts.slice(0, 6).join(',')}..`);
  }
});
