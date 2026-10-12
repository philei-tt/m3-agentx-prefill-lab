#!/usr/bin/env node
// Best net revenue per galaxy. Base: kernels at op efficiency 0.5, chunk 512 with fused-attention batching on the
// paged pool, host DRAM + SSD tiers, and a 62-stage decode ring on 16 decode galaxies with paged decode KV, decode tiers
// and queue backpressure (the page's ?opEff=0.5&chunk=512&batch=1&budget=8192&attn=fused&cache=pool&hostTier=1&
// decodeStages=62&decodeBatch=2&decodeCache=paging&decodeHostTier=1&decodeBackpressure=queue&decodeQueueMax=200).
// Varied:
//   prefill galaxies 4 / 6 / 8 ([2,4] stages, 4 per galaxy)
//   prefill token budget 4k / 8k / 16k / 32k
//   decode batch m = 1 / 2 / 4 / 8
//   decode-queue limit 100 / 145 / 200 / 300 per ring position (x m), or none
// Each configuration at its goodput point (p90 TTFT <= 10 s): net $/h (MiniMax's prices, output billed as decoded,
// $12 per galaxy-hour) over prefill + decode galaxies. There is no target on decode speed per user, so the tables
// show it (p10 / p50 tokens/s/u) next to each result.
// Usage: node tools/best_net_sweep.js [--workers 14] [--out results/best_net_sweep.json] > results/best_net_sweep.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const DG = SIM.COST.decodeGalaxies;
const BASE = {
  opEff: 0.5, chunk: 512, batch: true, attn: 'fused', cache: 'pool', hostTier: true, mesh: [2, 4], split: 'auto',
  decodeStages: 62, decodeCache: 'paging', decodeHostTier: true, decodeBackpressure: 'queue', decodeGalaxies: DG,
};
const GALAXIES = [4, 6, 8], BUDGETS = [4096, 8192, 16384, 32768], DBATCH = [1, 2, 4, 8], LIMITS = [100, 145, 200, 300, 0];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const jobs = [];
  for (const gx of GALAXIES) for (const budget of BUDGETS) for (const m of DBATCH) for (const L of LIMITS) {
    const cfg = Object.assign({}, BASE, { galaxies: gx, stages: 4 * gx, budget, decodeBatch: m, decodeQueueMax: L * m });
    jobs.push(pool.evalCfg(`${gx} ${budget} ${m} ${L}`, cfg, CONCS, SLO).then((r) => {
      const s = summarize(r.points, SLO), e = s.at ? SIM.economics(s.at, gx) : null;
      return { gx, budget, m, L, goodput: s.goodput, atEdge: s.atEdge, at: s.at, econ: e, perGx: e ? e.margin / (gx + DG) : -Infinity };
    }));
  }
  const rows = await Promise.all(jobs);
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  const pct = (x) => (Number.isFinite(x) ? (100 * x).toFixed(0) + '%' : '–');
  const lim = (r) => (r.L ? `${r.L} x ${r.m} = ${r.L * r.m}` : 'none');
  const line = (r) => {
    const p = r.at, e = r.econ;
    return `${usd(r.perGx).padStart(8)}/gx-h | net ${usd(e.margin)}/h on ${r.gx} + ${DG} gx | prefill ${r.gx} gx, budget ${r.budget / 1024}k | decode batch ${r.m}, queue limit ${lim(r)}`
      + ` | goodput ${k(r.goodput)}${r.atEdge ? ' (edge)' : ''} @C=${p.conc}, output ${k(p.outDecTps)} tok/s | decode p10/p50 ${Math.round(p.tsuP10)}/${Math.round(p.tsuP50)} tok/s/u`
      + ` | decode util ${pct(p.ringFullFrac)}, prefill busy ${pct(p.maxUtil)} | prefill-bound ${pct(p.boundPrefillFrac)}, decode-bound ${pct(p.boundDecodeFrac)}, balanced ${pct(p.boundBalancedFrac)}, demand-bound ${pct(p.boundDemandFrac)}`;
  };
  const ok = rows.filter((r) => r.econ).sort((a, b) => b.perGx - a.perGx);
  console.log(`== Net revenue per galaxy-hour at the goodput point (p90 TTFT <= ${SLO} s); ${ok.length} of ${rows.length} configurations meet the SLO`);
  console.log(`\n-- top 15`);
  for (const r of ok.slice(0, 15)) console.log('  ' + line(r));
  console.log(`\n-- best per prefill galaxy count`);
  for (const gx of GALAXIES) { const b = ok.find((r) => r.gx === gx); if (b) console.log('  ' + line(b)); }
  console.log(`\n-- best per decode batch`);
  for (const m of DBATCH) { const b = ok.find((r) => r.m === m); if (b) console.log('  ' + line(b)); }
  // decode speed per user has no target here: the best under a floor on the p10
  console.log(`\n-- best with decode p10 at or above a floor (tokens/s/u)`);
  for (const f of [20, 40, 60, 100]) { const b = ok.find((r) => r.at.tsuP10 >= f); console.log(`  p10 >= ${f}: ` + (b ? line(b) : 'none')); }
  console.log(`\n-- net $/galaxy-hour by prefill galaxies x decode batch (best budget and queue limit in each cell)`);
  console.log('  ' + ['gx \\ m', ...DBATCH.map((m) => `m=${m}`)].map((x) => x.padStart(9)).join(''));
  for (const gx of GALAXIES) console.log('  ' + [String(gx), ...DBATCH.map((m) => { const b = ok.find((r) => r.gx === gx && r.m === m); return b ? usd(b.perGx) : '–'; })].map((x) => x.padStart(9)).join(''));
  fs.writeFileSync(get('--out', path.join(RESULTS, 'best_net_sweep.json')), JSON.stringify({ slo: SLO, base: BASE, decodeGalaxies: DG,
    rows: rows.map((r) => ({ gx: r.gx, budget: r.budget, m: r.m, L: r.L, goodput: r.goodput, atEdge: r.atEdge, at: r.at, econ: r.econ, perGx: Number.isFinite(r.perGx) ? r.perGx : null })) }));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
