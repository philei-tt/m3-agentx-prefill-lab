#!/usr/bin/env node
// Decode KV layout, backpressure and batching, and their effect on revenue. Decode: a 62-stage ring with its KV memory
// modelled from the decode chips' DRAM (87 x 1M slots, 91M tokens of pages), decode-side host DRAM + SSD tiers on its 16 galaxies,
// speed on the measured M3 curve at 180 tokens/s/u @100k. Scenarios:
//   fixed slots, slot backpressure   today: a request waits for a free slot before prefill, holds it to decode end
//   fixed slots / paged, queue backpressure: prefill starts a request while the decode queue (requests holding slots
//                                    / pages, or parked) is under the limit (default 100); its KV goes
//                                    to free slots / pages, else to the decode SSDs, and a parked request takes slots /
//                                    pages after prefill (FIFO), reading its KV back first
//   paged, batched decode            m = 2 / 4 / 8 sessions per stage, decode-queue limit 100 x m (fixed slots cannot
//                                    batch: m x 62 sessions need more than 87 slots)
//   unlimited decode                 reference: no KV or ring limit
// Prefill: the decode-backpressure setup of README.md, 6 and 8 galaxies ([2,4] stages), paged pool + host DRAM + SSD
// tiers, today's kernels, round robin; best (by net revenue at the goodput point) of no batching (chunk 1024 / 2048)
// and batching (chunk 512 / 1024, budget 4k / 8k / 16k, 4 lanes). Goodput point: p90 TTFT <= 10 s (to the first
// decode token). Net revenue: MiniMax's
// prices, output tokens as decoded in the window, $12 per galaxy-hour, prefill + 16 decode galaxies.
// Usage: node tools/decode_layout_ab.js [--workers 14] [--out results/decode_layout_ab.json] > results/decode_layout_ab.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const RING = { decodeStages: 62, decodeSlots: 'auto', decodeHostTier: true, decodeGalaxies: SIM.COST.decodeGalaxies };
const DECODE = [
  ['fixed slots, slot backpressure (today)', Object.assign({ decodeCache: 'slots', decodeBackpressure: 'slot' }, RING)],
  ['fixed slots, queue backpressure (limit 100)', Object.assign({ decodeCache: 'slots', decodeBackpressure: 'queue' }, RING)],
  ['paged, queue backpressure (limit 100)', Object.assign({ decodeCache: 'paging', decodeBackpressure: 'queue' }, RING)],
  ...[2, 4, 8].map((m) => [`paged, batch ${m}, queue limit ${100 * m}`, Object.assign({ decodeCache: 'paging', decodeBackpressure: 'queue', decodeBatch: m, decodeQueueMax: 100 * m }, RING)]),
  ['(ref) unlimited decode', { decodeSlots: 0, decodeStages: 0 }],
];
const PREFILL = [];
for (const chunk of [1024, 2048]) PREFILL.push({ chunk, batch: false });
for (const chunk of [512, 1024]) for (const budget of [4096, 8192, 16384]) PREFILL.push({ chunk, batch: true, budget, lanesOverride: true, lanes: 4 });
const prefillTxt = (p) => (p.batch ? `chunk ${p.chunk}, batch ${p.budget / 1024}k` : `chunk ${p.chunk}, no batching`);

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const jobs = [];
  for (const gx of [6, 8]) for (const [label, dec] of DECODE) for (const pf of PREFILL) {
    const cfg = Object.assign({ galaxies: gx, stages: 4 * gx, mesh: [2, 4], split: 'auto', opEff: 0, cache: 'pool', hostTier: true }, pf, dec);
    jobs.push(pool.evalCfg(`${gx} ${label} ${prefillTxt(pf)}`, cfg, CONCS, SLO).then((r) => {
      const s = summarize(r.points, SLO), e = s.at ? SIM.economics(s.at, gx) : null;
      return { gx, label, dec, pf, goodput: s.goodput, at: s.at, econ: e, plan: r.plan };
    }));
  }
  const rows = await Promise.all(jobs);
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  const pct = (x) => (100 * x).toFixed(0) + '%';
  const res = { slo: SLO, concs: CONCS, decode: DECODE, prefill: PREFILL, galaxies: {} };
  for (const gx of [6, 8]) {
    console.log(`\n== ${gx} prefill galaxies (${4 * gx}x[2,4]) + 16 decode galaxies: best prefill config per decode layout, at its goodput point (p90 TTFT <= ${SLO} s)`);
    res.galaxies[gx] = [];
    for (const [label] of DECODE) {
      const all = rows.filter((r) => r.gx === gx && r.label === label);
      const best = all.filter((r) => r.econ).sort((a, b) => b.econ.margin - a.econ.margin)[0];
      res.galaxies[gx].push({ label, best: best && { pf: best.pf, goodput: best.goodput, at: best.at, econ: best.econ, plan: best.plan },
        all: all.map((r) => ({ pf: r.pf, goodput: r.goodput, conc: r.at && r.at.conc, margin: r.econ && r.econ.margin })) });
      if (!best) { console.log(`  ${label}: no point meets the SLO`); continue; }
      const p = best.at, e = best.econ;
      console.log(`  ${label}`);
      console.log(`    prefill ${prefillTxt(best.pf)} | goodput ${k(best.goodput)} @C=${p.conc} | p50/p90 TTFT ${p.ttftP50.toFixed(1)}/${p.ttftP90.toFixed(1)} s | ${Math.round(3600 * p.reqPerS).toLocaleString('en-US')} req/h`);
      console.log(`    revenue ${usd(e.revenue)}/h = input ${usd(e.inUsd)} + output ${usd(e.outUsd)} (${Math.round(p.outDecTps).toLocaleString('en-US')} tok/s decoded) | cost ${usd(e.cost)} | net ${usd(e.margin)}/h`);
      console.log(`    decode: held ${Math.round(p.decSlotsMean)} / max ${p.decSlotsMax}, decoding ${Math.round(p.decodingMean)} at ${Math.round(p.decodeTpsMean)} tokens/s/u (p10 ${Math.round(p.tsuP10)}, p50 ${Math.round(p.tsuP50)}), ring full ${pct(p.ringFullFrac)}`
        + ` | ${pct(p.decWaitPerS / p.reqPerS)} waited ${p.decWaitMean.toFixed(1)} s before prefill`
        + (p.decParkPerS > 0 ? ` | ${pct(p.decParkPerS / p.reqPerS)} parked on SSD (${Math.round(p.decParkedMean)} on average), decode SSD ${pct(p.decSsdUtil)} busy` : '')
        + (p.decSlotWaitFrac > 0 ? ` | ${pct(p.decSlotWaitFrac)} waited for slots / pages after prefill, first token +${(1e3 * p.decStartDelayMean).toFixed(0)} ms` : '')
        + (Number.isFinite(p.decHitRate) ? ` | decode-side hit ${(100 * p.decHitRate).toFixed(1)}%` : '')
        + (Number.isFinite(p.decPoolTokMean) ? ` | pool in use ${(p.decPoolTokMean / 1e6).toFixed(1)}M of ${(best.plan.decPoolTok / 1e6).toFixed(1)}M` : ''));
      console.log(`    every prefill config (net $/h at goodput): ${all.map((r) => `${prefillTxt(r.pf)} ${r.econ ? usd(r.econ.margin) : '–'}`).join('; ')}`);
    }
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'decode_layout_ab.json')), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
