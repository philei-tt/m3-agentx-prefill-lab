#!/usr/bin/env node
// Paths to positive net revenue at MiniMax-M3.1-Flash prices. Base: tools/best_net_sweep.js (chunk 512, fused-attention
// batching on the paged pool, host DRAM + SSD tiers, a 62-stage decode ring on 16 decode galaxies with paged decode KV,
// decode tiers and queue backpressure without a limit). Varied:
//   model     M3 / M3.1-Flash (dense layers ring-joint vs MSA)
//   opEff     0.5 / 1 (kernels halfway to / at the roofline targets)
//   prefill galaxies 4 / 6 / 8, budget 4k / 8k / 16k, decode batch m 8 / 16
// and, since decode turns out to bind, a decode grid on M3.1-Flash with roofline prefill kernels: decode speed per
// user decodeTps 180 (the target, the default) / 270 / 360 at 100k context, decode batch m 16 / 32, prefill galaxies
// 4 / 6 / 8, budget 4k / 8k.
// Each configuration at its goodput point (p90 TTFT <= 10 s): net $/h at M3.1-Flash's prices (SIM.PRICE) and at M3's,
// and, since revenue is linear in prices and cost in $/galaxy-hour, the break-even galaxy-hour cost (revenue / galaxies)
// and the break-even price multiple on M3.1-Flash's list price (cost / revenue).
// Usage: node tools/revenue_paths.js [--workers 14] [--out results/revenue_paths.json] > results/revenue_paths.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const DG = SIM.COST.decodeGalaxies, GXH = SIM.COST.galaxyUsdPerH;
const M3_PRICE = { inUsdPerM: 0.30, cachedUsdPerM: 0.06, outUsdPerM: 1.20 }; // minimax/minimax-m3, 2026-10-07
const BASE = {
  chunk: 512, batch: true, attn: 'fused', cache: 'pool', hostTier: true, mesh: [2, 4], split: 'auto',
  decodeStages: 62, decodeCache: 'paging', decodeHostTier: true, decodeBackpressure: 'queue', decodeQueueMax: 0, decodeGalaxies: DG,
};
const MODELS = ['m3', 'm3.1-flash'], OPEFF = [0.5, 1], GALAXIES = [4, 6, 8], BUDGETS = [4096, 8192, 16384], DBATCH = [8, 16];
const DEC_TPS = [180, 270, 360], DEC_BATCH = [16, 32], DEC_BUDGETS = [4096, 8192];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const jobs = [];
  const evalRow = (key, cfg) => pool.evalCfg(Object.values(key).join(' '), cfg, CONCS, SLO).then((r) => {
    const s = summarize(r.points, SLO);
    if (!s.at) return Object.assign({}, key, { goodput: s.goodput });
    const e = SIM.economics(s.at, key.gx), e3 = SIM.economics(s.at, key.gx, M3_PRICE);
    return Object.assign({}, key, { goodput: s.goodput, atEdge: s.atEdge, at: s.at, econ: e, econM3: e3, beGxUsd: e.revenue / (key.gx + DG), bePriceX: e.cost / e.revenue });
  });
  const decJobs = [];
  for (const tps of DEC_TPS) for (const m of DEC_BATCH) for (const gx of GALAXIES) for (const budget of DEC_BUDGETS) {
    const key = { model: 'm3.1-flash', opEff: 1, gx, budget, m, tps };
    decJobs.push(evalRow(key, Object.assign({}, BASE, { model: 'm3.1-flash', opEff: 1, galaxies: gx, stages: 4 * gx, budget, decodeBatch: m, decodeTps: tps })));
  }
  for (const model of MODELS) for (const opEff of OPEFF) for (const gx of GALAXIES) for (const budget of BUDGETS) for (const m of DBATCH) {
    jobs.push(evalRow({ model, opEff, gx, budget, m }, Object.assign({}, BASE, { model, opEff, galaxies: gx, stages: 4 * gx, budget, decodeBatch: m })));
  }
  const rows = (await Promise.all(jobs)).filter((r) => r.econ);
  const dec = (await Promise.all(decJobs)).filter((r) => r.econ);
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  const line = (r) => {
    const p = r.at, e = r.econ;
    return `net ${usd(e.margin).padStart(6)}/h (in ${usd(e.inUsd)} + out ${usd(e.outUsd)} - cost ${usd(e.cost)}) | at M3 prices ${usd(r.econM3.margin).padStart(5)}/h`
      + ` | break-even ${usd(r.beGxUsd)}/gx-h or ${r.bePriceX.toFixed(2)}x list | ${r.gx} + ${DG} gx, budget ${r.budget / 1024}k, decode batch ${r.m}`
      + ` | goodput ${k(r.goodput)}${r.atEdge ? ' (edge)' : ''} @C=${p.conc}, hit ${(100 * p.hitRate).toFixed(0)}%, output ${k(p.outDecTps)} tok/s, decode p10/p50 ${Math.round(p.tsuP10)}/${Math.round(p.tsuP50)} tok/s/u`;
  };
  const best = (f) => rows.filter(f).sort((a, b) => b.econ.margin - a.econ.margin)[0];
  console.log(`== Net revenue at MiniMax-M3.1-Flash prices ($${SIM.PRICE.inUsdPerM} / $${SIM.PRICE.cachedUsdPerM} / $${SIM.PRICE.outUsdPerM} per M input / cached / output),`
    + ` $${GXH} per galaxy-hour, ${DG} decode galaxies; goodput point (p90 TTFT <= ${SLO} s); ${rows.length} configurations meet the SLO`);
  console.log(`\n-- best per model x kernels (budget, galaxies and decode batch free)`);
  for (const model of MODELS) for (const opEff of OPEFF) { const b = best((r) => r.model === model && r.opEff === opEff); if (b) console.log(`  ${model.padEnd(10)} opEff ${opEff}: ` + line(b)); }
  console.log(`\n-- best per model x kernels x decode batch`);
  for (const model of MODELS) for (const opEff of OPEFF) for (const m of DBATCH) {
    const b = best((r) => r.model === model && r.opEff === opEff && r.m === m); if (b) console.log(`  ${model.padEnd(10)} opEff ${opEff} m=${String(m).padEnd(2)}: ` + line(b));
  }
  console.log(`\n-- net $/h at M3.1-Flash prices by model x kernels x prefill galaxies (best budget and decode batch)`);
  console.log('  ' + ['model, opEff', ...GALAXIES.map((g) => `${g} gx`)].map((x, i) => (i ? x.padStart(9) : x.padEnd(18))).join(''));
  for (const model of MODELS) for (const opEff of OPEFF) {
    console.log('  ' + [`${model}, ${opEff}`, ...GALAXIES.map((gx) => { const b = best((r) => r.model === model && r.opEff === opEff && r.gx === gx); return b ? usd(b.econ.margin) : '–'; })]
      .map((x, i) => (i ? x.padStart(9) : x.padEnd(18))).join(''));
  }
  console.log(`\n-- decode levers (M3.1-Flash, opEff 1): best per decode speed x decode batch (galaxies and budget free)`);
  const bestD = (f) => dec.filter(f).sort((a, b) => b.econ.margin - a.econ.margin)[0];
  for (const tps of DEC_TPS) for (const m of DEC_BATCH) { const b = bestD((r) => r.tps === tps && r.m === m); if (b) console.log(`  decode ${tps} tok/s/u @100k, m=${String(m).padEnd(2)}: ` + line(b)); }
  console.log(`\n-- net $/h by decode speed x prefill galaxies (M3.1-Flash, opEff 1; best budget and decode batch)`);
  console.log('  ' + ['decode tok/s/u', ...GALAXIES.map((g) => `${g} gx`)].map((x, i) => (i ? x.padStart(9) : x.padEnd(16))).join(''));
  for (const tps of DEC_TPS) console.log('  ' + [String(tps), ...GALAXIES.map((gx) => { const b = bestD((r) => r.tps === tps && r.gx === gx); return b ? usd(b.econ.margin) : '–'; })].map((x, i) => (i ? x.padStart(9) : x.padEnd(16))).join(''));
  fs.writeFileSync(get('--out', path.join(RESULTS, 'revenue_paths.json')), JSON.stringify({ slo: SLO, base: BASE, price: SIM.PRICE, priceM3: M3_PRICE, cost: SIM.COST,
    rows: rows.map((r) => ({ model: r.model, opEff: r.opEff, gx: r.gx, budget: r.budget, m: r.m, goodput: r.goodput, atEdge: r.atEdge, at: r.at, econ: r.econ, econM3: r.econM3, beGxUsd: r.beGxUsd, bePriceX: r.bePriceX })),
    decode: dec.map((r) => ({ tps: r.tps, gx: r.gx, budget: r.budget, m: r.m, goodput: r.goodput, atEdge: r.atEdge, at: r.at, econ: r.econ, econM3: r.econM3, beGxUsd: r.beGxUsd, bePriceX: r.bePriceX })) }));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
