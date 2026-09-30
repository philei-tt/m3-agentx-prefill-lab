#!/usr/bin/env node
// Why the infinite-cache hit rate of a replay window (~96.4%) is below the corpus ceiling (98.3%): which requests
// complete inside the measurement window, broken down by stream role, context size and trace position.
// Usage: node tools/hit_mix.js [--preset-study g4_k0] [--conc 744]
'use strict';
const SIM = require('../sim_core.js');
const { loadAll } = require('../run.js');
const { withFeatures } = require('../study.js');
const { DATA, STUDY } = require('../lib/paths.js');

const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const { TR, cal } = loadAll(DATA);
const R = require(STUDY).scenarios[get('--preset-study', 'g4_k0')];
const cfg = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra, { cache: 'inf', hostTier: false, laneArena: false, concurrency: Number(get('--conc', 744)), logRequests: true });
const res = SIM.simulate(TR, cal, cfg);
const B = 64, n = TR.req_blocks.length;
const ROLE = ['root', 'subagent', 'sibling', 'flat'];
// per request: role, context, trace position
const role = (r) => ROLE[TR.st_role[TR.req_stream[r]]] || '?';
const trEnd = (t) => (t + 1 < TR.tr_req0.length ? TR.tr_req0[t + 1] : n);
const pos = (r) => { const t = TR.req_trace[r], a = TR.tr_req0[t]; return (r - a) / Math.max(1, trEnd(t) - a - 1); };
const agg = (rs, key) => {
  const m = new Map();
  for (const r of rs) {
    const k = key(r), e = m.get(k) || { req: 0, tok: 0, hit: 0 };
    e.req++; e.tok += TR.req_blocks[r] * B; e.hit += TR.req_lcp_best[r] * B; m.set(k, e);
  }
  return m;
};
const all = Array.from({ length: n }, (_, i) => i).filter((r) => TR.req_blocks[r] > 0);
const win = res.reqLog || [];
const show = (title, key, order) => {
  const A = agg(all, key), W = agg(win, key);
  const tA = all.reduce((a, r) => a + TR.req_blocks[r], 0) * B, tW = win.reduce((a, r) => a + TR.req_blocks[r], 0) * B;
  console.log(`\n${title}: share of input tokens (corpus -> window) and infinite-cache hit rate`);
  for (const k of order) {
    const a = A.get(k), w = W.get(k); if (!a) continue;
    console.log(`  ${String(k).padEnd(14)} corpus ${(100 * a.tok / tA).toFixed(1).padStart(5)}% hit ${(100 * a.hit / a.tok).toFixed(1)}% | window ${w ? (100 * w.tok / tW).toFixed(1).padStart(5) + '% hit ' + (100 * w.hit / w.tok).toFixed(1) + '%' : '-'}`);
  }
};
const hit = (rs) => rs.reduce((a, r) => a + TR.req_lcp_best[r], 0) / Math.max(1, rs.reduce((a, r) => a + TR.req_blocks[r], 0));
console.log(`C=${cfg.concurrency}, ${win.length} requests completed in the window; infinite-cache hit: corpus ${(100 * hit(all)).toFixed(2)}%, window ${(100 * hit(win)).toFixed(2)}% (sim reports ${(100 * res.infHitRate).toFixed(2)}%)`);
show('by stream role', role, ROLE);
const ctxBin = (r) => { const c = TR.req_blocks[r] * B; return c < 64e3 ? '<64k' : c < 128e3 ? '64-128k' : c < 256e3 ? '128-256k' : c < 512e3 ? '256-512k' : '>=512k'; };
show('by context', ctxBin, ['<64k', '64-128k', '128-256k', '256-512k', '>=512k']);
const posBin = (r) => { const p = pos(r); return p < 0.25 ? 'first 25%' : p < 0.5 ? '25-50%' : p < 0.75 ? '50-75%' : 'last 25%'; };
show('by position in trace', posBin, ['first 25%', '25-50%', '50-75%', 'last 25%']);
