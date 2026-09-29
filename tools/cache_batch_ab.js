// today's 4-gx config + slot lanes/pool, with and without batching, and full paging: goodput and batch shape
const { Pool, summarize } = require('/data/philei/m3-agentx-prefill-lab/lib/pool.js');
const { CONCS, SLO } = require('/data/philei/m3-agentx-prefill-lab/study.js');
const base = { galaxies: 4, stages: 16, mesh: [2, 4], chunk: 2048, split: 'auto' };
const B = { batch: true, budget: 16384 };
const variants = [
  ['slots (today)', { cache: 'slots' }],
  ['slots + batch 16k', { cache: 'slots', ...B }],
  ['pool 4 lanes', { cache: 'pool', lanes: 4 }],
  ['pool 4 lanes + batch 16k', { cache: 'pool', lanes: 4, ...B }],
  ['pool 8 lanes + batch 16k', { cache: 'pool', lanes: 8, ...B }],
  ['paging', { cache: 'paging' }],
  ['paging + batch 16k', { cache: 'paging', ...B }],
  ['pool 4 lanes + host', { cache: 'pool', lanes: 4, hostTier: true }],
  ['pool 4 lanes + host + batch 16k', { cache: 'pool', lanes: 4, hostTier: true, ...B }],
  ['paging + host', { cache: 'paging', hostTier: true }],
  ['paging + host + batch 16k', { cache: 'paging', hostTier: true, ...B }],
  ['inf cache', { cache: 'inf' }],
  ['inf cache + batch 16k', { cache: 'inf', ...B }],
];
(async () => {
  const pool = new Pool(38);
  const res = await Promise.all(variants.map(([name, d]) => pool.evalCfg(name, Object.assign({}, base, d), CONCS, SLO).then((r) => [name, summarize(r.points, SLO)])));
  pool.close();
  for (const [name, s] of res) {
    const a = s.at;
    if (!a) { console.log(name, 'no passing point'); continue; }
    console.log(`${name.padEnd(32)} goodput ${(s.goodput / 1e3).toFixed(1).padStart(5)}k @C=${String(a.conc).padStart(4)} | req/batch ${a.avgSegsPerChunk.toFixed(2)} | chunk ${a.avgChunkTok.toFixed(0).padStart(5)} tok | pad ${(100 * a.padFrac).toFixed(0).padStart(2)}% | util ${(100 * a.maxUtil).toFixed(0).padStart(3)}% | hit ${(100 * a.hitRate).toFixed(1)}% (inf ${(100 * a.infHitRate).toFixed(1)}%) | reprefill ${(100 * a.reprefillFrac).toFixed(0)}% | p90 ${a.ttftP90.toFixed(1)} s`);
  }
})();
