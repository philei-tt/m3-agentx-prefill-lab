#!/usr/bin/env node
// Speculative decoding on the measured M3.1-Flash decode curves: revenue per galaxy-hour (prefill + decode galaxies) with the best prefill stacks of
// tools/stack_search.js (results/stack_search.json, opEff 1; 2 and 3 galaxies run the 4-galaxy stack), at the
// winner's budget and twice it, on 16 decode galaxies (paged decode KV + tiers, queue backpressure without a limit).
//   decode: tt-blaze main and the model card (unbatched, 62 stages), #5966 (16 users per 4x8 stage, 256 sessions).
//   spec:   off; batched verification at acceptance lengths (AL, tokens per cycle, bonus included) of
//           1 draft / 1.87 (M3's MTP at bf4 experts, accept rate 0.868), 3 / 1.49 (DeepSeek-V4 MTP, AgentX golden),
//           3 / 2.5, 3 / 3.0, 7 / 3.7 (Kimi-K2.5 EAGLE3, AgentX golden), each with the expert-weight share of a decode
//           step decodeMoeFrac 0.8 (the default) and 0.4; and pipelined verification (tt-blaze's MTP / DFlash today)
//           at 3 / 3.0. The drafter costs 10% of a plain step (specDraftFrac) and no galaxies.
// Usage: node tools/spec_sweep.js [--workers 14] [--out results/spec_sweep.json] > results/spec_sweep.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const DG = SIM.COST.decodeGalaxies, ALL_GX = [2, 3, 4, 6];
const DECODE = [
  { name: 'old main @87c36f5', cfg: { decodeStages: 62, decodeCurve: 'm31-main', decodeBatch: 1 } },
  { name: 'main (model card)', cfg: { decodeStages: 62, decodeCurve: 'm31-card', decodeBatch: 1, decodeSlots: 209 } },
  { name: '#5966 batched', cfg: { decodeStages: 256, decodeCurve: 'm31-b16', decodeBatch: 1 } },
];
const SPEC = [{ name: 'off', cfg: { specDraft: 0 } }];
for (const [n, al, src] of [[1, 1.87, 'M3 MTP'], [3, 1.49, 'DSv4 MTP'], [3, 2.5, ''], [3, 3.0, ''], [7, 3.7, 'K2.5 EAGLE3']]) {
  for (const f of [0.8, 0.4]) SPEC.push({ name: `batch ${n}/${al}${src ? ` (${src})` : ''}, moe ${f}`, cfg: { specDraft: n, specAL: al, specVerify: 'batch', decodeMoeFrac: f } });
}
SPEC.push({ name: 'pipe 3/3.0', cfg: { specDraft: 3, specAL: 3.0, specVerify: 'pipe' } });
const DEC_SIDE = { decodeCache: 'paging', decodeHostTier: true, decodeBackpressure: 'queue', decodeQueueMax: 0, decodeGalaxies: DG, specDraftFrac: 0.1 };

function toCfg(gx, opEff, st) {
  const c = Object.assign({}, st, { galaxies: gx, opEff });
  c.stages = gx * (c.stagesPerGx || 4); delete c.stagesPerGx;
  c.lanesOverride = c.cache === 'pool' && !c.laneArena && c.batch; if (c.lanesOverride) c.lanes = 4;
  return c;
}

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const prev = JSON.parse(fs.readFileSync(path.join(RESULTS, 'stack_search.json'), 'utf8'));
  const win = (gx) => prev.winners.find((w) => w.opEff === 1 && w.gx === (gx < 4 ? 4 : gx)).stack;
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const rows = await Promise.all(ALL_GX.flatMap((gx) => DECODE.flatMap((d) => SPEC.flatMap((s) => [1, 2].map((x) => {
    const st = win(gx), budget = st.budget * x;
    const cfg = Object.assign(toCfg(gx, 1, Object.assign({}, st, { budget })), DEC_SIDE, d.cfg, s.cfg);
    return pool.evalCfg(JSON.stringify(cfg), cfg, CONCS, SLO).then((r) => summarize(r.points, SLO))
      .then((sm) => ({ gx, dec: d.name, spec: s.name, budget, goodput: sm.goodput, at: sm.at, econ: sm.at ? SIM.economics(sm.at, gx) : null }));
  })))));
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  const best = (f) => rows.filter((r) => r.econ && f(r)).sort((a, b) => b.econ.marginPerGx - a.econ.marginPerGx)[0];
  console.log(`== Spec decode: revenue $ per galaxy-hour over prefill + decode galaxies (cost $${SIM.COST.galaxyUsdPerH}: above it is profit), best prefill stacks (opEff 1, best of budget x1 / x2), M3.1-Flash prices ($${SIM.PRICE.inUsdPerM} / $${SIM.PRICE.cachedUsdPerM} / $${SIM.PRICE.outUsdPerM}), $${SIM.COST.galaxyUsdPerH} per galaxy-hour, ${DG} decode galaxies`);
  console.log(`   spec: drafts / acceptance length (tokens per cycle); moe = expert-weight share of a decode step; drafter ${DEC_SIDE.specDraftFrac * 100}% of a step`);
  for (const d of DECODE) {
    console.log(`\n-- decode: ${d.name}`);
    console.log('  ' + ['spec', ...ALL_GX.map((g) => `${g} gx`), 'tok/s/u @128K', 'best: output tok/s'].map((x, i) => (i === 0 ? x.padEnd(40) : i <= ALL_GX.length ? x.padStart(9) : x.padStart(16))).join(''));
    for (const s of SPEC) {
      const tsu = SIM.decodeSpeed(Object.assign({}, SIM.DEFAULTS, DEC_SIDE, d.cfg, s.cfg), 131072);
      const b = best((r) => r.dec === d.name && r.spec === s.name);
      console.log('  ' + [s.name, ...ALL_GX.map((gx) => { const r = best((x) => x.dec === d.name && x.spec === s.name && x.gx === gx); return r ? usd(r.econ.revenuePerGx) : '–'; }),
        tsu.toFixed(0), b ? `${k(b.at.outDecTps)} @${b.gx} gx` : '–'].map((x, i) => (i === 0 ? x.padEnd(40) : i <= ALL_GX.length ? x.padStart(9) : x.padStart(16))).join(''));
    }
  }
  console.log(`\n-- best per decode x spec (by revenue per galaxy-hour)`);
  for (const d of DECODE) for (const s of SPEC) {
    const b = best((r) => r.dec === d.name && r.spec === s.name); if (!b) continue;
    const e = b.econ, p = b.at;
    console.log(`  ${(d.name + ', ' + s.name).padEnd(52)} revenue ${usd(e.revenuePerGx).padStart(6)}/gx-h, net ${usd(e.marginPerGx)}/gx-h (${usd(e.margin)}/h on ${e.galaxies} gx) | ${(e.cost / e.revenue).toFixed(2)}x list to break even`
      + ` | ${b.gx} + ${DG} gx, budget ${b.budget / 1024}k | goodput ${k(b.goodput)} @C=${p.conc}, output ${k(p.outDecTps)} tok/s, decode p10/p50 ${Math.round(p.tsuP10)}/${Math.round(p.tsuP50)} tok/s/u, prefill busy ${Math.round(100 * p.maxUtil)}%`);
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'spec_sweep.json')), JSON.stringify({ slo: SLO, decode: DECODE, spec: SPEC, decSide: DEC_SIDE, price: SIM.PRICE, cost: SIM.COST,
    rows: rows.map((r) => ({ gx: r.gx, dec: r.dec, spec: r.spec, budget: r.budget, goodput: r.goodput, at: r.at, econ: r.econ })) }));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
