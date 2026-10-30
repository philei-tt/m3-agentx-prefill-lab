#!/usr/bin/env node
// Best prefill stack for MiniMax-M3.1-Flash, then its net revenue under each decode scenario.
// 1) Stack search, per prefill galaxies (4 / 6 / 8) x kernels (opEff 0.5 / 1), decode unlimited: coordinate descent on
//    goodput (useful tok/s at p90 TTFT <= 10 s, finite cache, since tiering and paging are among the choices), from
//    the study's best stack (host DRAM + SSD tiers, pool arena, fused batching, tile padding, async handoff, MSA-local,
//    index_k stored once; 4 x [4,2] stages per galaxy). Each round tries every alternative of one dimension at a time
//    against the current best and keeps the best change; rounds repeat until nothing improves (at most 4).
//      topology [2,4] / [4,2] (4 stages per galaxy) / [4,4] (2); cache: pool with fixed lanes, arena 2M / 4M, full
//      paging; padding: tile even / tile owner / chunk 512 / chunk 2048; budget 4k-32k; on/off: async handoff,
//      MSA-local, index_k once, KV prefetch, host tiers; policy rr / srpt.
//    The winner is also run with cache 'inf' (peak useful tok/s, no SLO) for reference.
// 2) Revenue: each winner (and its budget halved / doubled) under the decode scenarios of tools/revenue_paths.js on
//    16 decode galaxies (paged decode KV + tiers, queue backpressure without a limit): the 180 tokens/s/u target at
//    decode batch 8 / 16 (modelled batching), measured bf4 unbatched and batched (#5966), and 270 / 360 at batch 16.
//    Net $/h at SIM.PRICE and SIM.COST at the goodput point. Decode-bound cells leave prefill idle, so the 4-galaxy
//    winners also run on 2 and 3 prefill galaxies ([4,4] stages: 4 / 6; [4,2]: 8 / 12).
// Usage: node tools/stack_search.js [--workers 14] [--out results/stack_search.json] > results/stack_search.txt
'use strict';
const fs = require('fs');
const path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');
const { RESULTS } = require('../lib/paths.js');
const { usd } = require('../lib/price.js');

const DG = SIM.COST.decodeGalaxies;
const GALAXIES = [4, 6, 8], SMALL = [2, 3], ALL_GX = SMALL.concat(GALAXIES), OPEFF = [0.5, 1];
const START = {
  model: 'm3.1-flash', mesh: [4, 2], split: 'auto', cache: 'pool', laneArena: true, arenaTokens: 4e6, hostTier: true,
  batch: true, attn: 'fused', budget: 16384, reqPad: 'tile', placement: 'even', asyncHandoff: true, msaLocal: true,
  idxDerep: true, prefetchKV: false, policy: 'rr', decodeStages: 0,
};
// one dimension = a list of alternative patches; `stagesPerGx` sets the stage count from the galaxies
const DIMS = {
  topo: [{ mesh: [2, 4], stagesPerGx: 4 }, { mesh: [4, 2], stagesPerGx: 4 }, { mesh: [4, 4], stagesPerGx: 2 }],
  cache: [{ cache: 'pool', laneArena: false }, { cache: 'pool', laneArena: true, arenaTokens: 2e6 }, { cache: 'pool', laneArena: true, arenaTokens: 4e6 }, { cache: 'paging' }],
  pad: [{ reqPad: 'tile', placement: 'even' }, { reqPad: 'tile', placement: 'owner' }, { reqPad: 'chunk', chunk: 512 }, { reqPad: 'chunk', chunk: 2048 }],
  budget: [4096, 8192, 16384, 32768].map((budget) => ({ budget })),
  async: [{ asyncHandoff: true }, { asyncHandoff: false }],
  msa: [{ msaLocal: true }, { msaLocal: false }],
  idx: [{ idxDerep: true }, { idxDerep: false }],
  prefetch: [{ prefetchKV: false }, { prefetchKV: true }],
  host: [{ hostTier: true }, { hostTier: false }],
  policy: [{ policy: 'rr' }, { policy: 'srpt' }],
};
const DECODE = [
  { name: '180 target, batch 8 (modelled)', cfg: { decodeStages: 62, decodeTps: 180, decodeCurve: 'm3', decodeBatch: 8 } },
  { name: '180 target, batch 16 (modelled)', cfg: { decodeStages: 62, decodeTps: 180, decodeCurve: 'm3', decodeBatch: 16 } },
  { name: 'bf4 unbatched (main, measured)', cfg: { decodeStages: 62, decodeTps: 154, decodeCurve: 'm3', decodeBatch: 1 } },
  { name: 'bf4 batched 4x8 x16 (#5966)', cfg: { decodeStages: 256, decodeTps: 56, decodeCurve: 'flat', decodeBatch: 1 } },
  { name: '270 @100k, batch 16', cfg: { decodeStages: 62, decodeTps: 270, decodeCurve: 'm3', decodeBatch: 16 } },
  { name: '360 @100k, batch 16', cfg: { decodeStages: 62, decodeTps: 360, decodeCurve: 'm3', decodeBatch: 16 } },
];
const DEC_SIDE = { decodeCache: 'paging', decodeHostTier: true, decodeBackpressure: 'queue', decodeQueueMax: 0, decodeGalaxies: DG };

// a stack (galaxies + patches) -> simulator config; fixed pool lanes with batching need an explicit lane count
function toCfg(gx, opEff, st) {
  const c = Object.assign({}, st, { galaxies: gx, opEff });
  c.stages = gx * (c.stagesPerGx || 4); delete c.stagesPerGx;
  c.lanesOverride = c.cache === 'pool' && !c.laneArena && c.batch; if (c.lanesOverride) c.lanes = 4;
  return c;
}
const desc = (st) => `${st.stagesPerGx === 2 ? '2' : '4'}x[${st.mesh}]/gx, ${st.cache === 'paging' ? 'paging' : st.laneArena ? `arena ${st.arenaTokens / 1e6}M` : 'fixed lanes'}`
  + `, ${st.reqPad === 'tile' ? `tile ${st.placement}` : `chunk ${st.chunk}`}, budget ${st.budget / 1024}k`
  + `${st.asyncHandoff ? ', async' : ''}${st.msaLocal ? ', MSA-local' : ''}${st.idxDerep ? ', idx once' : ''}${st.prefetchKV ? ', prefetch' : ''}${st.hostTier ? ', tiers' : ''}, ${st.policy}`;

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const pool = new Pool(Number(get('--workers', 14)));
  const t0 = Date.now();
  const memo = new Map();
  const goodput = (cfg, slo = SLO) => {
    const id = JSON.stringify([cfg, slo]);
    if (!memo.has(id)) memo.set(id, pool.evalCfg(id, cfg, CONCS, slo).then((r) => summarize(r.points, slo)));
    return memo.get(id);
  };
  // 1) coordinate descent per scenario
  const search = async (gx, opEff) => {
    let best = Object.assign({ stagesPerGx: 4 }, START), bestG = (await goodput(toCfg(gx, opEff, best))).goodput;
    const trail = [`start ${Math.round(bestG)}`];
    for (let round = 0; round < 4; round++) {
      const tries = [];
      for (const [dim, alts] of Object.entries(DIMS)) for (const p of alts) {
        const st = Object.assign({}, best, p);
        if (JSON.stringify(st) === JSON.stringify(best)) continue;
        tries.push(goodput(toCfg(gx, opEff, st)).then((s) => ({ dim, st, g: s.goodput || 0 })));
      }
      const rs = (await Promise.all(tries)).sort((a, b) => b.g - a.g);
      if (!rs.length || rs[0].g <= bestG * 1.002) break;
      best = rs[0].st; bestG = rs[0].g; trail.push(`${rs[0].dim} -> ${Math.round(bestG)}`);
    }
    const inf = await goodput(Object.assign(toCfg(gx, opEff, best), { cache: 'inf', hostTier: false }), 1e6);
    return { gx, opEff, stack: best, goodput: bestG, infPeak: inf.peak ? inf.peak.usefulTps : NaN, trail };
  };
  const winners = await Promise.all(GALAXIES.flatMap((gx) => OPEFF.map((o) => search(gx, o))));
  // 2) revenue under each decode scenario, the winner's budget and its neighbours
  const runs = winners.concat(winners.filter((w) => w.gx === 4).flatMap((w) => SMALL.map((gx) => Object.assign({}, w, { gx }))));
  const rows = await Promise.all(runs.flatMap((w) => DECODE.flatMap((d) => [w.stack.budget / 2, w.stack.budget, w.stack.budget * 2]
    .filter((b) => b >= 2048 && b <= 65536).map((budget) => {
      const cfg = Object.assign(toCfg(w.gx, w.opEff, Object.assign({}, w.stack, { budget })), DEC_SIDE, d.cfg);
      return goodput(cfg).then((s) => ({ gx: w.gx, opEff: w.opEff, dec: d.name, budget, goodput: s.goodput, at: s.at, econ: s.at ? SIM.economics(s.at, w.gx) : null }));
    }))));
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1) + 'k';
  console.log(`== Best M3.1-Flash prefill stacks (decode unlimited; goodput at p90 TTFT <= ${SLO} s, finite cache; peak = cache 'inf', no SLO)`);
  for (const w of winners) console.log(`  ${w.gx} gx, opEff ${w.opEff}: goodput ${k(w.goodput)} (inf peak ${k(w.infPeak)}) | ${desc(w.stack)} | ${w.trail.join(', ')}`);
  const best = (f) => rows.filter((r) => r.econ && f(r)).sort((a, b) => b.econ.margin - a.econ.margin)[0];
  console.log(`\n== Net $/h with those stacks (2 and 3 gx: the 4 gx stack): M3.1-Flash prices ($${SIM.PRICE.inUsdPerM} / $${SIM.PRICE.cachedUsdPerM} / $${SIM.PRICE.outUsdPerM}), $${SIM.COST.galaxyUsdPerH} per galaxy-hour, ${DG} decode galaxies (best budget)`);
  for (const o of OPEFF) {
    console.log(`\n-- opEff ${o}`);
    console.log('  ' + ['decode', ...ALL_GX.map((g) => `${g} gx`)].map((x, i) => (i ? x.padStart(9) : x.padEnd(34))).join(''));
    for (const d of DECODE) console.log('  ' + [d.name, ...ALL_GX.map((gx) => { const b = best((r) => r.opEff === o && r.dec === d.name && r.gx === gx); return b ? usd(b.econ.margin) : '–'; })].map((x, i) => (i ? x.padStart(9) : x.padEnd(34))).join(''));
  }
  console.log(`\n-- best per decode scenario`);
  for (const d of DECODE) {
    const b = best((r) => r.dec === d.name); if (!b) continue;
    const e = b.econ, p = b.at;
    console.log(`  ${d.name.padEnd(34)} net ${usd(e.margin).padStart(7)}/h (in ${usd(e.inUsd)} + out ${usd(e.outUsd)} - cost ${usd(e.cost)}) | break-even ${usd(e.revenue / (b.gx + DG))}/gx-h or ${(e.cost / e.revenue).toFixed(2)}x list`
      + ` | ${b.gx} + ${DG} gx, opEff ${b.opEff}, budget ${b.budget / 1024}k | goodput ${k(b.goodput)} @C=${p.conc}, output ${k(p.outDecTps)} tok/s, decode p10/p50 ${Math.round(p.tsuP10)}/${Math.round(p.tsuP50)} tok/s/u, prefill busy ${Math.round(100 * p.maxUtil)}%`);
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'stack_search.json')), JSON.stringify({ slo: SLO, start: START, dims: DIMS, decode: DECODE, decSide: DEC_SIDE, price: SIM.PRICE, cost: SIM.COST, winners,
    rows: rows.map((r) => ({ gx: r.gx, opEff: r.opEff, dec: r.dec, budget: r.budget, goodput: r.goodput, at: r.at, econ: r.econ })) }));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
