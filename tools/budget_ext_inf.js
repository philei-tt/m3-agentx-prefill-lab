// Infinite cache: extend the budget beyond 32k (the roofline-kernel optimum was at the grid edge)
const { Pool, summarize } = require('/data/philei/m3-agentx-prefill-lab/lib/pool.js');
const { withFeatures, CONCS, SLO } = require('/data/philei/m3-agentx-prefill-lab/study.js');
const study = require('/data/philei/m3-agentx-prefill-lab/results/study.json');
(async () => {
  const pool = new Pool(38);
  const jobs = [];
  for (const [key, R] of Object.entries(study.scenarios)) {
    const keys = R.bestKeys.filter((k) => !['var', 'fused', 'batch'].includes(k));
    const g = R.grid[0].extra;
    const base = Object.assign(withFeatures(R.base, keys), { stages: g.stages, mesh: g.mesh, replicas: g.replicas, cache: 'inf', hostTier: false, laneArena: false, batch: true });
    for (const B of [32768, 49152, 65536]) {
      jobs.push(pool.evalCfg(`${key} fixed ${B}`, Object.assign({}, base, { layout: 'fixed', chunk: 128, attn: 'seq', budget: B }), CONCS.concat([5120, 6144]), SLO).then((r) => ({ key, v: 'fixed C=128', B, s: summarize(r.points, SLO) })));
      jobs.push(pool.evalCfg(`${key} var ${B}`, Object.assign({}, base, { layout: 'var', chunk: 5120, attn: 'fused', budget: B }), CONCS.concat([5120, 6144]), SLO).then((r) => ({ key, v: 'var fused', B, s: summarize(r.points, SLO) })));
    }
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const r of res.sort((a, b) => a.key.localeCompare(b.key) || a.v.localeCompare(b.v) || a.B - b.B))
    console.log(`${r.key} ${r.v.padEnd(12)} B=${r.B / 1024}k: ${(r.s.goodput / 1e3).toFixed(1)}k @C=${r.s.at && r.s.at.conc} req/chunk ${r.s.at && r.s.at.avgSegsPerChunk.toFixed(2)} p90 ${r.s.at && r.s.at.ttftP90.toFixed(1)}${r.s.atEdge ? ' [edge]' : ''}`);
})();
