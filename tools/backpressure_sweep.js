#!/usr/bin/env node
// Decode backpressure strategies on the study's best prefill configurations (today's kernels, 4 and 8 galaxies), with
// and without prefill batching. Decode: a 62-stage ring with modelled KV memory (145 x 1M slots), decode host DRAM + SSD
// tiers, 180 tokens/s/u @100k on the M3 curve.
//   decode: slot backpressure on fixed slots at decode batch 1 (today) / 2 | queue backpressure, limit 100 / 145 /
//           200 x decode batch m, on fixed slots at m = 1 / 2 and on paged decode KV at m = 1 / 2 / 4 / 8
//   prefill batching: on (the best config's fused, tile-padded batches; budget 8k / 16k) | off (per-request
//           attention, a 2048 / 5120-token pass)
// Each cell keeps the prefill budget with the higher net revenue at the goodput point (p90 TTFT <= 10 s); MiniMax's
// prices, output billed as decoded, $12 per galaxy-hour, prefill + 16 decode galaxies.
// Usage: node tools/backpressure_sweep.js [--workers 14] [--out results/backpressure_sweep.json] > results/backpressure_sweep.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const { RESULTS, STUDY } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const RING = { decodeStages: 62, decodeSlots: 'auto', decodeHostTier: true, decodeGalaxies: SIM.COST.decodeGalaxies };
// queue limits per ring position (x decode batch m): below, at and above the 145 modelled slots
const LIMITS = [100, 145, 200];
const DECODE = [
  ['slot backpressure, fixed slots (today)', { decodeCache: 'slots', decodeBackpressure: 'slot' }],
  ['slot backpressure, fixed slots, decode batch 2', { decodeCache: 'slots', decodeBackpressure: 'slot', decodeBatch: 2 }],
  ...LIMITS.flatMap((L) => [
    [`queue backpressure (limit ${L} x m), fixed slots`, { decodeCache: 'slots', decodeBackpressure: 'queue', decodeQueueMax: L }],
    [`queue backpressure (limit ${L} x m), fixed slots, decode batch 2`, { decodeCache: 'slots', decodeBackpressure: 'queue', decodeBatch: 2, decodeQueueMax: 2 * L }],
    ...[1, 2, 4, 8].map((m) => [`queue backpressure (limit ${L} x m), paged, decode batch ${m}`, { decodeCache: 'paging', decodeBackpressure: 'queue', decodeBatch: m, decodeQueueMax: L * m }]),
  ]),
];
const PREFILL = [
  ['batched', [8192, 16384].map((b) => ({ batch: true, attn: 'fused', lanesOverride: true, budget: b }))],
  ['unbatched', [2048, 5120].map((b) => ({ batch: false, attn: 'request', lanesOverride: false, budget: b }))],
];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const jobs = [];
  for (const sk of ['g4_k0', 'g8_k0']) {
    const R = study.scenarios[sk], best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    for (const [pfName, pfs] of PREFILL) for (const pf of pfs) for (const [label, dec] of DECODE) {
      const cfg = Object.assign({}, best, pf, RING, dec);
      jobs.push(pool.evalCfg(`${sk} ${pfName} ${pf.budget} ${label}`, cfg, CONCS, SLO).then((r) => {
        const s = summarize(r.points, SLO);
        return { sk, gx: best.galaxies, pfName, budget: pf.budget, label, goodput: s.goodput, at: s.at, econ: s.at ? SIM.economics(s.at, best.galaxies) : null };
      }));
    }
  }
  const rows = await Promise.all(jobs);
  pool.close();
  const pct = (x) => (Number.isFinite(x) ? Math.round(100 * x) + '%' : '–');
  const out = { slo: SLO, decode: DECODE, prefill: PREFILL, cells: [] };
  for (const sk of ['g4_k0', 'g8_k0']) {
    const R = study.scenarios[sk], g = R.grid[0].extra;
    console.log(`\n== ${R.label}: best stack ${g.stages}x[${g.mesh}], decode 62 stages / modelled memory; net $/h at the goodput point (p90 TTFT <= ${SLO} s)`);
    for (const [label] of DECODE) for (const [pfName] of PREFILL) {
      const cands = rows.filter((r) => r.sk === sk && r.label === label && r.pfName === pfName && r.econ);
      const b = cands.sort((x, y) => y.econ.margin - x.econ.margin)[0];
      if (!b) { console.log(`  ${label} | prefill ${pfName}: no point meets the SLO`); continue; }
      const p = b.at, e = b.econ;
      out.cells.push({ sk, label, pfName, budget: b.budget, goodput: b.goodput, at: p, econ: e });
      console.log(`  ${label} | prefill ${pfName} (${b.budget}): net ${usd(e.margin)}/h (revenue ${usd(e.revenue)}) | goodput ${(b.goodput / 1e3).toFixed(1)}k @C=${p.conc}, ${(3.6 * p.reqPerS).toFixed(1)}k req/h, `
        + `output ${(p.outDecTps / 1e3).toFixed(1)}k tok/s | decode p10/p50 ${Math.round(p.tsuP10)}/${Math.round(p.tsuP50)} tok/s/u | decode utilization ${pct(p.ringFullFrac)} | prefill busy ${pct(p.maxUtil)}, `
        + `waits on decode ${pct(p.pfStarvedSlotFrac)} | prefill-bound ${pct(p.boundPrefillFrac)}, `
        + `decode-bound ${pct(p.boundDecodeFrac)} (memory ${pct(p.boundDecodeMemoryFrac)}, queue full ${pct(p.boundDecodeQueueFrac)}), balanced ${pct(p.boundBalancedFrac)}, demand-bound ${pct(p.boundDemandFrac)}`);
    }
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'backpressure_sweep.json')), JSON.stringify(out));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
