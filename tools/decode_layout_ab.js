#!/usr/bin/env node
// Decode KV layout and backpressure, and their effect on revenue. Decode: a 62-stage ring with 86 x 1M slots of KV
// memory (today's code fits 86-89; earlier runs used 75), decode-side host DRAM + SSD tiers on its 16 galaxies,
// speed on the measured M3 curve at 180 tokens/s/u @100k. Scenarios:
//   fixed slots, slot backpressure   today: a request waits for a free slot before prefill, holds it to decode end
//   fixed slots, queue backpressure  a free slot if there is one, else its KV is parked on the decode SSDs and it takes
//                                    a slot after prefill (FIFO), reading its KV back first; no limit / 32 parked
//   paged                            the whole memory a paged pool; every session past prefill decodes at once
//   unlimited decode                 reference: no KV or ring limit
// Prefill: the decode-backpressure setup of README.md, 6 and 8 galaxies ([2,4] stages), paged pool + host DRAM + SSD
// tiers, today's kernels, round robin; best (by net revenue at the goodput point) of no batching (chunk 1024 / 2048)
// and batching (chunk 512 / 1024, budget 4k / 8k / 16k, 4 lanes). Goodput point: p90 TTFT <= 10 s (to the first
// decode token) and, with --tsu T (default 50), 90% of decode sessions at >= T tokens/s/u. Net revenue: MiniMax's
// prices, output tokens as decoded in the window, $12 per galaxy-hour, prefill + 16 decode galaxies.
// Usage: node tools/decode_layout_ab.js [--workers 14] [--tsu 50] [--out results/decode_layout_ab.json] > results/decode_layout_ab.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const RING = { decodeStages: 62, decodeSlots: 86, decodeHostTier: true, decodeGalaxies: SIM.COST.decodeGalaxies };
const DECODE = [
  ['fixed slots, slot backpressure (today)', Object.assign({ decodeCache: 'slots', decodeBackpressure: 'slot' }, RING)],
  ['fixed slots, queue backpressure, no limit', Object.assign({ decodeCache: 'slots', decodeBackpressure: 'queue' }, RING)],
  ['fixed slots, queue backpressure, 32 parked', Object.assign({ decodeCache: 'slots', decodeBackpressure: 'queue', decodeQueueMax: 32 }, RING)],
  ['paged: 86M-token pool', Object.assign({ decodeCache: 'paging' }, RING)],
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
  const tsuMin = Number(get('--tsu', 50));
  const t0 = Date.now();
  const jobs = [];
  for (const gx of [6, 8]) for (const [label, dec] of DECODE) for (const pf of PREFILL) {
    const cfg = Object.assign({ galaxies: gx, stages: 4 * gx, mesh: [2, 4], split: 'auto', opEff: 0, cache: 'pool', hostTier: true }, pf, dec);
    jobs.push(pool.evalCfg(`${gx} ${label} ${prefillTxt(pf)}`, cfg, CONCS, SLO, 4, { tsuMin }).then((r) => {
      const s = summarize(r.points, SLO, tsuMin), e = s.at ? SIM.economics(s.at, gx) : null;
      return { gx, label, dec, pf, goodput: s.goodput, at: s.at, econ: e, plan: r.plan };
    }));
  }
  const rows = await Promise.all(jobs);
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  const pct = (x) => (100 * x).toFixed(0) + '%';
  const res = { slo: SLO, tsuMin, concs: CONCS, decode: DECODE, prefill: PREFILL, galaxies: {} };
  for (const gx of [6, 8]) {
    console.log(`\n== ${gx} prefill galaxies (${4 * gx}x[2,4]) + 16 decode galaxies: best prefill config per decode layout, at its goodput point (p90 TTFT <= ${SLO} s${tsuMin > 0 ? `, p10 decode speed >= ${tsuMin} tokens/s/u` : ''})`);
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
        + (p.decLaneWaitFrac > 0 ? ` | ${pct(p.decLaneWaitFrac)} waited for a slot after prefill, first token +${(1e3 * p.decStartDelayMean).toFixed(0)} ms` : '')
        + (Number.isFinite(p.decHitRate) ? ` | decode-side hit ${(100 * p.decHitRate).toFixed(1)}%` : '')
        + (Number.isFinite(p.decPoolTokMean) ? ` | pool in use ${(p.decPoolTokMean / 1e6).toFixed(1)}M of ${(best.plan.decPoolTok / 1e6).toFixed(1)}M` : ''));
      console.log(`    every prefill config (net $/h at goodput): ${all.map((r) => `${prefillTxt(r.pf)} ${r.econ ? usd(r.econ.margin) : '–'}`).join('; ')}`);
    }
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'decode_layout_ab.json')), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
