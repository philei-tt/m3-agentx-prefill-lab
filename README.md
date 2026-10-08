# MiniMax-M3 prefill: AgentX traffic roofline

Traffic-level model of the M3 pipeline-prefill system on Blackhole galaxies. It replays the **AgentX 1M corpus** the way
AIPerf's `inferencex-agentx-mvp` scenario does and runs every request through a **per-op roofline** calibrated to the
16-stage hardware runs of #57827. The goal is to rank features and hyperparameters by what they do for the whole system
on realistic traffic, not by single-cell benchmarks.

* Repository: https://github.com/philei-tt/m3-agentx-prefill-lab (also mirrored in tt-metal branch `philei/m3-traffic-sim`, `models/demos/minimax_m3/traffic_sim/`)
* Hosted copy of the web app: https://claude.ai/artifact/3Yresb3qbWJQpvpQMabaq6
* Tracking issue: https://github.com/tenstorrent/tt-metal/issues/57827

## Quick start: run the web app

Needs only Node.js ≥ 18 (no `npm install`, no dependencies). The traffic, calibration and study results are in the repository.

```bash
git clone https://github.com/philei-tt/m3-agentx-prefill-lab && cd m3-agentx-prefill-lab
node server.js                      # builds dist/index.html on first start, serves http://127.0.0.1:8765
node server.js --port 9000          # another port; --host 0.0.0.0 to listen on all interfaces
```

On a remote machine, forward the port from your laptop and open http://localhost:8765:

```bash
ssh -L 8765:localhost:8765 <remote-host>     # then, on the remote host: node server.js
```

The page is self-contained: the simulator runs in your browser (Web Workers), so the server only serves one file.
Fonts come from Google Fonts when reachable, with system fallbacks otherwise.
**Copy link** (Explore tab) gives a URL such as `http://localhost:8765/?galaxies=8&batch=1&budget=5120#explore` holding every setting that differs from the defaults; opening it loads that configuration and runs the sweep.
**Metrics at** (Explore tab, or a click on a chart) shows the tiles (useful tok/s, TTFT, hit rate, input tokens and requests per hour, wasted compute, requests per batch) at any concurrency instead of the goodput point. A concurrency the sweep did not run is simulated on demand; it is added to the charts but never moves goodput. The link keeps it as `at=<concurrency>`.
`node server.js --rebuild` forces a rebuild; at startup the server also rebuilds when any input is newer than `dist/index.html`.

## Metric

**Goodput = useful tok/s at p90 TTFT ≤ 10 s**, maximised over concurrency. The sweep and the goodput rule live in one file, `lib/sweep.js`, used by the study and inlined into the web page:
* **Grid:** an ascending concurrency grid, stopped early once well past the knee.
* **Extension:** the grid is extended ×1.5 (up to 3 times) while its top point still meets the SLO. A result that still passes at the top is flagged as a lower bound, shown as "≥" on the page.
* **Cliff bisection:** the cache cliff is steep (neighbouring grid points can differ by 10%+), so the sweep bisects the SLO crossing 4 times in log concurrency, without re-running any point.
* **Interpolation:** when the first failing point has higher throughput, goodput is interpolated to the SLO crossing.

A point where no request finished (TTFT undefined) never passes. Useful tokens per request are capped at the tokens it actually prefilled.
*Useful* tokens are the tokens an infinite prefix cache would still have to prefill (`in - 64·lcp_best`). Re-prefilled tokens
(evicted, misaligned, never materialised) and padding count as processed but not useful. Each result also reports the same
configuration with an infinite cache, TTFT p50/p90, the hit rate against the ∞-cache hit rate, and the split of processed
tokens into useful, re-prefill and padding. At the goodput point it also reports hourly volume (`hourly()` in `sim_core.js`): input tokens per hour of the requests completed, split into new (prefilled, re-prefill included) and cached (prefix hit), requests completed per hour, and the revenue
those input tokens bill per hour.

**Revenue per hour** prices new tokens (cache misses, re-prefill included) at the input price and prefix hits at the
cache-read price. The defaults (`SIM.PRICE` in `sim_core.js`) are MiniMax's own endpoint for `minimax/minimax-m3` on
OpenRouter (provider "Minimax", Oct 7 2026): **$0.30 / M input, $0.06 / M cache read, $1.20 / M output**, no cache-write
charge. That is also the model-level price and most providers' (the cheapest third-party provider is about 20% lower). Override it with `--price-in` / `--price-cached` (USD per
million tokens) on `run.js`, `analyze.js` and `tools/feature_table.js`, or with the Revenue & cost fields on the web page.
* **Output tokens are not counted in revenue.** This system only prefills; decode generates the output tokens, so their
  revenue ($1.20 / M) counts only in the margin below, next to the decode galaxies' cost. AgentX requests average about
  1.1k output tokens against 120k–150k input tokens at the goodput points, so output adds roughly 10–15% to a bill.
* **A better cache bills less per token** (a hit costs a fifth of a miss); it raises revenue only through the extra
  requests the same hardware can then serve.

**Net revenue per hour** (the page's Net Revenue tile, `economics()` in `sim_core.js`) is input + output revenue of the requests completed per hour, minus
(prefill galaxies + decode galaxies) × USD per galaxy-hour. Defaults (`SIM.COST`): **$12 per galaxy-hour** (a rough
operating cost) and **16 decode galaxies**, one M3 decode instance of 64 sessions, so pair it with `decodeSlots: 64`
(unlimited slots overstate what 16 decode galaxies serve). Each run reports `outTps`, the output tokens of the requests
completed in the window; study points from before it have no margin. Override with `--price-out`, `--galaxy-usd` and
`--decode-galaxies` on `run.js`, or the Revenue & cost fields on the page.

## Files

| file | what |
|---|---|
| `sim_core.js` | **The whole model** (no dependencies): per-op roofline + calibration, stage plan and memory, KV residency (static slots / paged pool with lanes / host DRAM + SSD offload tiers / paging / ∞), and the closed-loop replay. The artifact inlines it and node requires it. |
| `prep_traffic.py` | Corpus → `traffic.bin` + `traffic.json`: agent-chain split, DAG, end-to-start delays, spawn/join, replay barriers, prefix-tree pieces. Takes 23 s on 4 cores. |
| `collect_calib.py` | Measured data → `calib_data.json` (committed): zone profiles, per-rank per-chunk-position medians of runs A/B/C, matrix tables. |
| `validate.js` | Calibration report: fitted efficiencies, per-rank stage times, and all 105 matrix cells, model vs measured. |
| `run.js` | One configuration or a concurrency sweep from the CLI (`--preset`, `--set key=value`, `--conc`, `--price-in` / `--price-cached`). |
| `study.js` | Greedy feature roadmap, leave-one-out, topology/budget/lane grid and sensitivity, over {4, 8} galaxies × {today's kernels, roofline kernels}. |
| `analyze.js`, `tools/study_detail.js` | Print study results. |
| `lib/pool.js`, `lib/sweep.js` | Worker-thread pool; the concurrency sweep and the goodput rule (shared with the page). |
| `lib/scope.js` | Complexity bin and one-line summary per feature (Roadmap and README tables). |
| `tests/` | `node tests/test_sweep.js`, `node tests/test_model.js` (or `npm test`): sweep/goodput rule and cost-model/replay regression checks. |
| `build_artifact.js`, `artifact/template.html` | Build the single-file page: the core, calibration, 4 MB of traffic as base64, the study summary and the presets. `--standalone` wraps it as a full HTML document for `server.js`. |
| `server.js`, `package.json` | Zero-dependency local web server (see Quick start); `npm start` / `npm run build` / `npm run study` are shortcuts. |
| `feature_details.js` | The per-feature explanations shown when a feature is expanded in the Roadmap table. |
| `lib/price.js` | `--price-in` / `--price-cached` / `--price-out` / `--galaxy-usd` / `--decode-galaxies` overrides of `SIM.PRICE` and `SIM.COST` for the CLIs. |
| `lib/paths.js` | Where data and results are read from: env `M3SIM_DATA` / `M3SIM_RESULTS`, else `./data` and `./results`, else the exabox scratch copies. |
| `on_node.sh` | `JOB=<slurm id> ./on_node.sh <cmd>` runs on the compute node with soft ulimits raised to the hard limits. |
| `tools/` | Helpers: `feature_table.js` (README tables), `study_detail.js`, `grid_by_topology.js`, `dump_cells.js` (per-cell stage medians), `traffic_stats.js`, `smoke.sh` (every feature path), `investigate.sh`. |

Data:
* `data/traffic.bin` + `traffic.json`: the preprocessed corpus, 5 MB, derived from the Apache-2.0 HF dataset.
* `data/zones/`: `parse_zone_perf.py` outputs of the per-op profiles, the inputs to `collect_calib.py`.
* `results/study.json`: the full study, which the page's Roadmap tab and presets are built from.

`dist/` (the built page) is not committed.

## How to run the model and the study

Any machine with Node ≥ 18 works. On exabox use a cpu_only Slurm allocation, never the login node; `JOB=<id> ./on_node.sh <cmd>` runs a command there with raised ulimits.

```bash
node validate.js                                                  # calibration report (model vs 105 measured cells)
node run.js --preset today-C --conc 16,64,256                     # one config over a few concurrencies
node run.js --study-preset g8_k0 --set decodeTps=90 --conc 512,1024
node study.js --workers 36                                        # full study -> results/study.json (about 23 min with `--workers 13` on 14 cores; about 8 min on 36 threads before the offload tiers)
node analyze.js && node tools/feature_table.js                    # print results / README tables
node tools/feature_table.js --price-in 0.6 --price-cached 0.06     # revenue at other prices
node run.js --preset today-C --set decodeSlots=64 --galaxy-usd 12 --decode-galaxies 16   # margin per hour
node build_artifact.js --standalone                               # rebuild dist/index.html (server.js does this itself)
# regenerate the inputs (needs numpy, the AgentX traces and the #57827 run directories):
python3 prep_traffic.py --traces <traces.jsonl> --procs 32        # data/traffic.bin + traffic.json
python3 collect_calib.py --matrix <run dir with runA/runB/runC>   # calib_data.json
```

One simulation of 1800 s of traffic at C=1024 takes 0.3–0.5 s. The full study (about 500 configurations and several thousand runs with cliff refinement) takes about 8 minutes on 36 threads, 12 on a 14-core laptop.

## Traffic replay (AIPerf `inferencex-agentx-mvp` semantics)

The replay rules below were ported from the AIPerf source (`ai-dynamo/aiperf` @ `c78644090ff`: `timing/strategies/agentic_replay.py`, `timing/trajectory_source.py`, `timing/branch_orchestrator.py`, `timing/replay_dependencies.py`, `dataset/loader/weka_trace.py`, `dataset/loader/weka_agent_chains.py`). `prep_traffic.py` reproduces AIPerf's streams, spawn/join/offsets, replay scopes and barrier predecessors exactly on all 393 traces (checked against AIPerf's own loader code), and the sim's t* snapshots match `TrajectorySource._snapshot_for`.

* **Corpus.** `semianalysisai/cc-traces-weka-062126`, full variant: 393 sessions, 98,827 requests, 21.6B input tokens, contexts up to 990k. The ∞-cache hit rate is 98.3%.
* **Lanes.** `concurrency` = N lanes, each replaying one tree: the root chain plus its child streams. Children are the sub-agents, each split into its own agent chains by hash-id LCP detection, and the flattened agent chains that the same detection splits off the top-level requests (`WEKA_SPLIT_FLATTENED_AGENTS`, 49% of top-level requests). 9,843 streams in all.
  * **Start.** t* is uniform between the first and the last recorded request start (AgentX uses a 0–1 ratio). Every stream live at t* resumes at its first request at/after t* and gets a primer: its last request before t*, as a real prefill with `max_tokens=1`. A branch that starts after t* is spawned later by its spawn turn. When t* is after the last root start the lane is rootless (its children drain, then it recycles). Profiling starts when all primers are done.
  * **Recycle.** A drained tree recycles to the next session, sequentially, from turn 0 with a new cache-bust. Trees share nothing.
* **Delays.** Each next request waits `max(0, t_k − t_{k−1} − api_time_{k−1})` after the live end of the previous one. The delay is **not capped**. Root-chain gaps have a median of 4.5 s, a mean of 285 s and a p99 of 51 min. The only cap is AIPerf's 10 s system-idle cap. `gapCap` exists as a knob, default off.
* **Children.** Sub-agents (and, separately, flat chains) that share a spawn turn and a join turn form one branch. The branch starts at its earliest marker; it overlaps its spawn turn (dispatch at the spawn turn's issue instead of its return) when that start is before the spawn turn's recorded end. Each child's first request is offset from the spawn turn's issue (overlap) or from the branch start. The join turn fires immediately once the last child ends.
* **Barriers.** AIPerf's cross-stream replay barriers: within a replay scope (the root with its flat chains, or one sub-agent with its chains) a request waits, on every other stream, for the latest request that had completed at its recorded start. 62% of requests have such predecessors (76k edges, `pred_head`/`pred_list` in `traffic.bin`, optional). Requests before t* count as completed; primers are not gated.
* **Decode.** A request ends at prefill done + `out / decodeTps` (default 180 tok/s).
* **Decode slots (backpressure).** `decodeSlots` (default 0 = unlimited) caps the requests decode holds KV for. A request
  takes a slot before it may start prefill (waiting in FIFO order, inside its TTFT, while all are held) and frees it when
  decode ends, so prefill only runs requests decode has room for. Taking the slot at admission, not at hand-off, is
  what tt-d-gen does: it needs the decode slot up front to start KV migration eagerly. Every run reports the slots held (mean and max over
  the window), how many of them are decoding, and the share of requests that waited and their mean wait. M3 decode
  today holds about 64 sessions, one per pipeline stage (tt-blaze #4220); batched decode (m = 8) targets about 504.
* **Decode concurrency.** `decodeConcurrency` (default 0 = unlimited) caps the sessions decode generates for at once,
  separately from the KV slots: a 64-stage decode ring carries one session per stage, while its memory holds about 85
  slots of 1M tokens per stage (tt-blaze: K/V sharded by head over the 4 mesh rows and replicated over the 2 columns,
  index-K split over the columns, bf8, 340 B per token per chip). A request whose prefill is done waits for a decode
  position in FIFO order, holding its slot; runs report the share that waited and the mean wait.
* **Why prefill starves.** Runs report the share of the window prefill's first stage had nothing to issue (stage
  free, nothing queued, no started request with tokens left), by cause: `pfStarvedSlotFrac` (requests were waiting
  for a decode KV slot), `pfStarvedDecodeFrac` (none were, but requests that finished prefill were waiting for a
  decode position, so their sessions could not send their next request), `pfStarvedIdleFrac` (no demand). Separately,
  `sendBlockFrac` is the share of time a stage is held after its compute by the synchronous handoff to the next stage
  (about 25% of a saturated stage; `asyncHandoff` removes it).
* **Window.** The profiling window is 1800 s.

**Consequence:** an AgentX lane is mostly idle. Saturating a pipeline therefore takes hundreds to thousands of lanes. The KV working set then grows with concurrency, and throughput is limited by a **cache cliff**: past it, evictions turn into re-prefill, which raises TTFT, which leaves more KV idle and evicted.

## Cost model

Each layer is decomposed into the ops the implementation runs.

* **MoE/MSA layer:** 2× norm all-gather, qkv, index branch, all-gathers of the K/V and index_k prefix, indexer, sparse SDPA, o_proj + RS, shared expert (+RS), router, dispatch, experts, combine, moe_reduce (+RS), misc.
* **Dense layer:** ring-joint SDPA = max(compute ∝ T·(k+T/2), gather of the valid prefix ∝ kv_len/SP) + a fixed cost per call, plus the MLP and its CCLs. The op bounds the gather to [0, kv_len) since tt-metal #47539 (M3 passes `kv_actual_isl` and `logical_n`). Only the gather *buffer* is capacity-sized. `boundedDense: false` models the old whole-lane gather.

**Roofline.** Each op gets a roofline time from FLOPs, DRAM bytes and ethernet bytes on the stage mesh. The Blackhole numbers come from tt-moe-nappkin `lib/system.py`: 608/304 TF (LoFi/HiFi2), 512 GB/s, 100/50 GB/s CCL bi/uni.

**Calibration.**
1. **Zone profiles.** `eff = roofline / (zone time − latency floor)`, using the zone profiles (chunk 5120, 51k cached) for [2,4], [8,4] and [4,2]. [4,4] is the geometric midpoint of [2,4] and [8,4].
2. **Pipeline fit.** A fit on 7,840 per-rank, per-chunk-position medians of the 16×[2,4] runs A/B/C produces:
   * an MoE multiplier of about 1.06 at chunk 5120 and 1.10 at 2048 (pipeline vs single-stage profile, 2D fabric);
   * dense ring-joint: compute 33%, kv_len gather 1.35% (about 100 ms per 1M context tokens per chunk, which dominates at chunk 2048), and 2.75 ms fixed per call.
     * In runs A/B/C the capacity is cached + 51,200, so a capacity term and a kv term look alike across cells.
     * A joint regression on the single-dense-layer stages settles it: about 210 ms per 1M kv vs 19 ms per 1M capacity at chunk 5120, and 100 vs 2 at 2048.
     * Before Sep 29, the fit used a whole-capacity scan. That made today's 1M slots look far too slow and produced a phantom ×3 "bounded dense gather" feature.
   * embedding 1.5 ms;
   * a blocking send of 18 ms at 5120 and 7.5 ms at 2048;
   * a hop latency of 16 ms at 5120 and 14 ms at 2048.

   Two findings came out of the fit:
   * **Padded chunk tails are cheaper.** Routed MoE ops use *actual* tokens (the `actual_isl` trim), so a 640-token request in a 5120 chunk is 18% cheaper than a full chunk.
   * **Today's expert kernel behaves like weight read + compute,** not the max of the two.
3. **Torus rings (`torus`).** A galaxy is two 4×4 tori. Every measurement used line (2D mesh) collectives, so the fitted efficiencies are line efficiencies. On an axis that runs as a ring, the link terms are scaled by textbook ratios:
   * all-gather/reduce-scatter ×(N−1)/N;
   * all-to-all (dispatch/combine, var-layout KV write) ×0.5;
   * the ring-joint KV pass ×0.5;
   * the collective latency floor ×0.5.

   There are three modes. `'full'` (default) gives rings only to [4,4] stages, i.e. a whole torus. `'axes'` also gives them to any 4-long axis spanning a torus row or column: [2,4] TP, [4,2] SP, [8,4] TP. `'off'` is the measured setup. `tools/torus_ab.js` compares the modes on every topology.
4. **Validation.** Replaying all 105 matrix cells gives |error| of 5.5% median, 11.0% p90 and 15.9% max (loaded new tok/s and idle TTFT). The old capacity-scan fit scored 4.2 / 8.9 / 15.4%, but for the wrong reason; the dense stages themselves now fit 3× better (log rms 0.033 vs 0.100). The largest misses are the deepest cells, 7–10% under on the dense stages at 553k cached. Run `validate.js` to see it.

**`opEff` knob.** It moves each op geometrically from its measured efficiency to a target (70% matmul, 80% DRAM/link, overlapped expert weight reads) and lowers the latency floors. 0 means today's kernels; 1 means roofline kernels.

**Batching.** Batching works with every cache. With static slots a request holds its 1M slot for its whole trip through the pipeline (today's `slot_id`), so a full pipeline needs (requests per batch) × (stages) slots: the chunk units per batch with the fixed layout, budget / (32·SP) with the variable layout, 1 without batching. `makePlan` reports out of memory when that exceeds the slots that fit. At 4 galaxies, 16 × [2,4] stages hold 20 slots, so batching on slots does not fit; 4 × [8,4] stages fit up to 4 requests per batch. In the study, whose base config uses slots, `batch` requires `pool`. Several requests share one chunk: MoE and projections run on the padded total, and attention runs per request (`seq`) or once per chunk (`fused`), with a 110-core wave-quantization factor. Variable layout pads each segment to 32·SP tokens and adds an all-to-all KV write per layer.

**Memory per stage.**
* weights: bf4 experts, bf8 attention, bf16 shared/dense MLP, embedding on stage 0, LM head on the last stage;
* `reserveGB` per chip plus activation buffers;
* KV: 1088 B/token/layer for K/V (bf8), plus index_k at 128 × (2 B bf16 | 1.0625 B bf8) × (TP replicas | 1).

KV capacity is the minimum over stages. At 4 galaxies with bf16 index_k replicated ×4, that is about 21.4M tokens, or 20 static 1M slots.

Every buffer that must hold a whole request (slots, fixed lanes, the lane arena) must be at least 990,016 tokens, the largest AgentX request; `makePlan` rejects smaller ones.

## KV residency

* **`slots`** (today): one 1M slot per stream; LRU over idle slots. The hit is the prefix shared with the stream's previous request, resumed at any 32-token boundary (`unaligned`, default on since tt-metal #57636 merged; `--set unaligned=false` floors it to a chunk multiple, the old behaviour).
  * With `hostTier` (the offload tiers, below): a reclaimed slot's KV (its stream's latest request) is written to host DRAM over PCIe, and the new occupant waits for that write-back; host DRAM overflows to SSD. The stream's next request reads back the prefix it shares with it before admission (or at start, if its slot was reclaimed while it was queued). One copy per stream, LRU per tier, index_k stored once. In the Oct 7 study, at 4 galaxies with today's kernels, the tiers take goodput from 3.4k to 16.4k (p90 6.6 s, hit 94.8%), and adding the pool (one lane per stage) takes it to 25.2k. The tiers do not require the pool, and they are the first roadmap step in every scenario.
* **`pool`**: lanes (fixed 1M per stage, or request-sized in an arena) plus a content-addressed paged pool with LRU over prefix-tree pieces.
  * **Lanes per stage** are derived: with a per-stage lane table a stage works on one chunk at a time, so it needs one lane per request in the batch it is processing (1 without batching, the chunk units per batch with fixed-layout batching), times the buffers of `copyMode` (sequential 1, double 2, overlap3 3). A global lane table (`laneScope: 'global'`) holds a lane for the request's whole trip through the pipeline, so it derives stages × requests per batch, and it allows only sequential copies (buffering a lane reserved for the whole trip gains nothing). `lanesOverride: true` sets the count to `lanes` instead; it is allowed only with batching on the pool. Variable-layout batching has no chunk units, so it must override. Every request in a batch holds its own lane, so the count caps the requests per batch, under every policy. Each extra 1M lane comes out of the pool: at 4 galaxies without the SSD tier, one lane beats three by 2–4% and eight lanes cost 13–22% with batching. The study's batched stacks override (its pool feature uses 4 lanes; the grid tries 2/4/6); its unbatched pool steps derive 1 lane.
  * The prefix tree is compressed into pieces cut at branch points and request ends, so every request touches whole pieces. LRU over pieces matches LRU over 64-token pages except that a piece is evicted whole. Against a brute-force page LRU, total hits agree within 0.15%.
  * The hit is looked up when the request becomes ready (which refreshes it in the LRU, but does not pin it), and re-checked when the request starts. Pages evicted meanwhile are recomputed; pages demoted to host meanwhile are fetched over PCIe before the first chunk.
  * The new KV enters the pool when the lane is freed: with per-stage lanes, when stage 0 finishes the request's last chunk (later stages follow in FIFO order); with global lanes, at prefill completion.
  * Copy-in/out is DRAM-bound per stage. By default (`copyMode: 'sequential'`) the full copy time is charged on the stage. `'double'` overlaps copies with compute and charges 25% of their time for DRAM contention; the study's pool feature pins this mode, which was the default when `results/study.json` was computed. `'overlap3'` is static triple buffering.
  * With `hostTier` (the **KV offload tiers**), device evictions move to host DRAM, host DRAM evicts to SSD, SSD drops (exclusive LRU tiers over prefix-tree pieces). Hits are read back before admission. Offloaded copies store index_k once (no TP replicas; re-broadcast on fetch).
  * Known approximation: KV fetched from host at READY is staged on device without being counted against capacity until the request starts.
* **`paging`**: an ideal paged kernel (no lanes, no copies). The pages a request is writing are reserved in the pool from start to completion. The offload tiers also work behind paging.
* **KV offload tiers** (`hostTier`): device DRAM → host DRAM → SSD, per galaxy host. Defaults, from the [Tenstorrent Galaxy Blackhole Server User Guide v1.8](https://docs.tenstorrent.com/_downloads/3086863c42126fd0d63b01baccf8432e/galaxy-blackhole.pdf) unless noted:
  * **Host DRAM** `hostDramGBPerGalaxy` 576 GB (6 × 96 GB DDR5-4800 RDIMM). The share left for KV is derived in `makePlan` (`plan.hostBudget`), and these reserves are estimates:

    | reserve | default | key |
    |---|---|---|
    | headroom | 10% (58 GB) | `hostHeadroom` |
    | OS and services | 16 GB | `hostOsGB` |
    | model runtime (tt-metal, inference server, program caches) | 32 GB | `hostRuntimeGB` |
    | staging for KV migration to decode | 32 GB | `hostKvStagingGB` |
    | pinned PCIe host memory: 1 GiB hugepage channel per chip (tt-metal/UMD picks min(4, chips per MMIO device) = 1 on a Galaxy, where every chip has its own PCIe link) | 34 GB | `hostPinnedGBPerChip` |
    | this galaxy's share of the model weights (host copies while loading / reloading) | 33 GB at 8 galaxies, 67 GB at 4 | `hostStageWeights` |

    That leaves about **371 GB of KV per galaxy at 8 galaxies** (37M tokens per pipeline at 80.6 KB/token, next to 56M of device pool) and 337 GB at 4 galaxies (17M tokens, next to 20M). Set `hostDramGBPerGalaxy: 0` to drop the host tier. The AgentX rules cap host DRAM per system; check that cap against this budget.
  * **SSD** `ssdTBPerGalaxy` 32 TB (4 E1.S bays; shipped systems carry 4 × 7.68 TB Samsung PM9D3a, MZTL67T6HBLC = 30.7 TB): 3.2B tokens at 8 galaxies. Bandwidth `ssdReadGBsPerGalaxy` 31.5, `ssdWriteGBsPerGalaxy` 27.2 GB/s: the [PM9D3a](https://semiconductor.samsung.com/ssd/datacenter-ssd/pm9d3a/) reads up to 12,000 MB/s and writes up to 6,800 MB/s at PCIe 5.0 x4, but Rev C chassis run the E1.S links at Gen4 x4 (a BIOS setting in the Exabox provisioning runbook), which caps a read at 7.88 GB/s per drive. Reads and writes share the drives.
  * **PCIe** `pcieGBsPerGalaxy` 63 GB/s each way, full duplex (fetches and write-backs do not share it). Per tray, one chip has a Gen5 x8 host link and seven have x1 links that train at Gen4 (user guide: "PCIe Gen5 1x8 and 7x1"; Quanta S7TK product spec: "7x1 PCIe G4"). KV is spread evenly over the chips and each chip moves its own share over its own link, so the x1 chips set the pace: 32 × 1.97 GB/s. Relaying through each tray's x8 chip over Ethernet would allow up to 4 × (31.5 + 7 × 1.97) = 181 GB/s; no software does that today.
  * Not modelled: host DRAM bandwidth (6 channels of DDR5-4800, about 230 GB/s, above the links' combined peak), SSD endurance and per-IO latency. A device page counts as free as soon as its write-back is queued. Runs report `hostReadTps`, `ssdReadTps` and the busy share of each link (`pcieH2DUtil`, `pcieD2HUtil`, `ssdUtil`).
* **`inf`**: infinite cache.

## Scheduling (`policy`)

* **`rtc`** (used by the study and the configs built from it; formerly `fcfs`, which is still accepted): run to completion, oldest first. A started request is never preempted: every chunk continues the oldest started request; a new request starts only when no started request has tokens left. With batching, started requests are continued first and new ones fill the leftover room.
* **`srpt`**: also run to completion, with the waiting queue sorted shortest-new-first (30 s aging). Despite the name it does not preempt.
* **`rr`** (default): round robin, as in tt-d-gen (`PrefillQueue` + `PrefillWriter::step`). A request joins the back of the queue when it is admitted, i.e. when it gets a slot or lane. Each turn the front request takes one chunk and goes to the back if it has chunks left. With batching, requests are popped from the front and each takes as many of the batch's remaining chunk units as it can fill (all it has left, if that is less), as one attention call. Unfinished requests go to the back behind everything waiting, so a request is never split into two runs within one batch. On static slots the in-flight cap is tt-d-gen's ChunkFifo, max(8, 4 × slots).
  * **`batchMaxChunks`** (default 0 = no limit) caps the chunk units one request takes in one batch. With no limit a long request goes through the pipeline in budget-sized passes, so its last pass is large and has to clear every stage before its first token; with 1 it moves one chunk per pass, as without batching, and the budget fills from other requests.
  * **On the pool** (`cache: 'pool'`): a *lane* is a per-stage KV slot that the attention kernels run on (1M tokens, or request-sized in an arena). A request takes a lane for each turn, and every turn copies the segment's new KV out to the pool. The partial KV stays pinned in the pool until the request finishes, so a lane never holds the only copy and can be handed to another request at any turn. The lane count bounds the requests per batch, not the requests in progress.
  * The lane count is the same as for the other policies (see **Lanes per stage** under KV residency). A request takes a lane for each turn, so the count caps the requests per batch but not the requests in progress.
  * Copy-in: a request whose lane still holds its context, i.e. no other request has used that lane since its last turn, skips the copy-in. On a miss it takes a free lane (an empty one first, else the least recently used) and copies its whole context so far in. Arena lanes always copy in. Results report `rrLaneReuse` (share of continuation turns that skipped the copy-in) and `rrCopyInTps`. Copying in every turn instead made no measurable difference to goodput (≤0.5%, even with sequential copies), so it is not an option.
  * Round robin on the pool, and paging under rr, pin the cached prefix of every request in progress. A request is admitted only while the pool can hold all in-progress requests in full; until then it waits in the queue. This matches tt-d-gen, where admission needs a free slot, a full pool queues the request rather than rejecting it, and in-flight slots are never evicted.

## Findings (study of Oct 7 2026: KV offload tiers (host DRAM + 32 TB SSD per galaxy) behind static slots, round-robin scheduling, sequential pool copies, derived lane counts, unaligned resume in the baseline; decode 180 tok/s, AIPerf-exact replay, `results/study.json`)

Goodput in useful tok/s at p90 TTFT ≤ 10 s. "Today" = 16×[2,4] (or 32×[2,4]), chunk 2048, auto split, static 1M slots, round-robin scheduling, unaligned resume.

**Changes from the previous study (Oct 1):**
* **The KV offload tiers replace the 1 TB SSD tier.** The Oct 1 study had one 1 TB tier per galaxy at 64 GB/s, shared by reads and writes. Now device evictions go to host DRAM (about 337 GB of KV per galaxy at 4 galaxies, 371 GB at 8) over PCIe (63 GB/s each way), and host DRAM evicts to a 32 TB SSD (31.5 / 27.2 GB/s read / write). Today's baselines have no tier and are unchanged.
* **The best grid configs rise ×1.35–1.79:** 45.1k → 60.7k, 82.6k → 129k, 104k → 150k, 161k → 289k. The greedy full stacks rise 39.8k / 78.3k / 94.8k / 158k → 48.6k / 122k / 131k / 277k.
  * Re-prefill in the full stack drops from 18–59% of prefilled tokens to 2–4%. The best configs reach 88–99% of their own ∞-cache goodput (Oct 1: 40–75%).
* **The tiers are still the first step everywhere,** now ×4.0–9.9 when added (Oct 1: ×3.7–6.7) and ×2.2–2.8 leave-one-out (×1.6–1.9).
* **With capacity no longer binding, compute features gain.** Batching becomes P0 (×1.44 / ×1.63 at 8 gx today), variable chunk P1 (×1.14 / ×1.17) and the arena P1 (×1.08 leave-one-out at 8 gx today, but ×0.91 at 4 gx today). index_k stored once and bf8 drop to P2: ×1.00–1.01 with today's kernels, though still ×1.04–1.17 with roofline kernels, where tier bandwidth binds.
* **At 4 gx with roofline kernels the best topology is 4×[8,4]** (129.0k vs 125.3k for 16×[4,2]). The other three best configs are [4,2] stages, as before.
* The sensitivity runs now vary the tiers: no host DRAM tier, 8 TB SSD per galaxy, PCIe 181 GB/s per galaxy (x8 relay), half SSD bandwidth (they used to vary 0.5 / 2 TB and 16 / 256 GB/s of SSD).

Earlier studies are kept on exabox under `/data/philei/m3_traffic_sim/results/`, which is not in git. The tables below are printed by `node tools/feature_table.js`.

| scenario | today | greedy full stack | best grid config | best config with ∞ cache |
|---|---|---|---|---|
| 4 galaxies, today's kernels | 3.4k | 48.6k | **60.7k** (16×[4,2], 6 lanes, budget 16k) | 62.4k |
| 4 galaxies, roofline kernels | 4.8k | 122k | **129k** (4×[8,4], 2M arena, budget 16k) | 146k |
| 8 galaxies, today's kernels | 11.1k | 131k | **150k** (32×[4,2], 2M arena, budget 16k) | 151k |
| 8 galaxies, roofline kernels | 15.0k | 277k | **289k** (32×[4,2], 2M arena, budget 4k) | 326k |

The ∞-cache column re-runs each best config with an infinite cache, so it moves with the config: at 4 gx with roofline kernels the Oct 1 best config (16×[4,2]) reached 209k with an infinite cache, the new [8,4] one 146k.

Hourly volume and revenue at the goodput point (the simulated concurrency where each configuration reaches its goodput, p90 TTFT ≤ 10 s): input tokens of the requests completed per hour, split into new tokens the pipeline prefilled (re-prefill included) and cached prefix hits, requests completed per hour, and the revenue of those input tokens (output tokens not counted).

Revenue at $0.30/M input, $0.06/M cached.

| scenario | configuration | C | input tok/h | new tok/h | cached tok/h | hit | requests/h | revenue/h |
|---|---|---|---|---|---|---|---|---|
| 4 galaxies, today's kernels | today | 40 | 469.6M | 59.9M | 409.8M | 87.3% | 3.1k | $42.55 |
| 4 galaxies, today's kernels | greedy full stack | 496 | 4.89B | 175.5M | 4.72B | 96.4% | 38.5k | $336 |
| 4 galaxies, today's kernels | best grid config | 672 | 6.25B | 226.0M | 6.02B | 96.4% | 48.2k | $429 |
| 4 galaxies, roofline kernels | today | 56 | 594.3M | 146.5M | 447.8M | 75.4% | 4.2k | $70.81 |
| 4 galaxies, roofline kernels | greedy full stack | 1256 | 12.99B | 448.1M | 12.54B | 96.6% | 102.0k | $887 |
| 4 galaxies, roofline kernels | best grid config | 1320 | 13.54B | 479.0M | 13.06B | 96.5% | 107.0k | $928 |
| 8 galaxies, today's kernels | today | 104 | 1.06B | 200.3M | 864.4M | 81.2% | 9.1k | $112 |
| 8 galaxies, today's kernels | greedy full stack | 1496 | 13.71B | 488.5M | 13.22B | 96.4% | 107.9k | $940 |
| 8 galaxies, today's kernels | best grid config | 1704 | 15.37B | 565.0M | 14.80B | 96.3% | 122.3k | $1,058 |
| 8 galaxies, roofline kernels | today | 128 | 1.42B | 342.7M | 1.07B | 75.8% | 12.3k | $167 |
| 8 galaxies, roofline kernels | greedy full stack | 2968 | 29.90B | 1.02B | 28.88B | 96.6% | 230.3k | $2,038 |
| 8 galaxies, roofline kernels | best grid config | 3072 | 31.25B | 1.06B | 30.19B | 96.6% | 241.2k | $2,129 |

**Decode backpressure** (`decodeSlots`, not part of the study). One configuration: 8 galaxies, 32 stages, chunk 512,
batching with an 8k budget, paged pool, today's kernels; goodput at p90 TTFT ≤ 10 s, revenue at $0.45 / $0.09 per M.
Slots held count requests from admission to prefill until the end of their decode.

| decode | slots | goodput | C | requests/h | revenue/h | slots held, mean / max | decoding | waited for a slot |
|---|---|---|---|---|---|---|---|---|
| 180 tok/s | unlimited | 30.7k | 312 | 25.2k | $372 | 75 / 135 | 40 | – |
| 180 tok/s | 128 | 30.7k | 312 | 25.3k | $372 | 75 / 128 | 40 | 1%, 0.3 s |
| 180 tok/s | 64 | 28.6k | 280 | 23.1k | $338 | 58 / 64 | 36 | 46%, 3.9 s |
| 180 tok/s | 32 | 16.1k | 144 | 12.1k | $151 | 27 / 32 | 20 | 57%, 3.4 s |
| 100 tok/s | unlimited | 28.2k | 320 | 22.8k | $342 | 97 / 157 | 64 | – |
| 100 tok/s | 128 | 28.0k | 312 | 22.7k | $335 | 93 / 128 | 63 | 5%, 0.9 s |
| 100 tok/s | 64 | 23.0k | 248 | 18.1k | $254 | 62 / 64 | 49 | 68%, 3.8 s |

One 64-session decode keeps up with this prefill at 180 tok/s for a 7% loss, and at 100 tok/s for 18%. 128 slots
is enough either way.

Margin per hour of the same runs, at $12 per galaxy-hour, 16 decode galaxies per 64 slots, and either OpenRouter's
prices ($0.30 / $0.06 / $1.20 per M input / cached / output) or 1.5× them ($0.45 / $0.09 / $1.80):

| decode | slots | galaxies (prefill + decode) | cost/h | revenue/h, OpenRouter (in + out) | margin | revenue/h, 1.5× | margin |
|---|---|---|---|---|---|---|---|
| 180 tok/s | 64 | 8 + 16 | $288 | $225 + $28 = $253 | −$35 | $338 + $43 = $381 | +$93 (24%) |
| 100 tok/s | 64 | 8 + 16 | $288 | $169 + $21 = $190 | −$98 | $254 + $32 = $286 | −$2 |
| 180 tok/s | 128 | 8 + 32 | $480 | $248 + $31 = $279 | −$201 | $372 + $47 = $419 | −$61 |

Decode is two thirds of the cost but bills only the output tokens: with a slot held from admission, about 40% of the
64 slots hold requests still queued or in prefill rather than decoding.

Each cell below is "G #step / LOO":
* **G** is the gain at the greedy step where the feature was added (the step number is the build order);
* **LOO** is the leave-one-out loss when the feature is removed from the full stack.

Order and tier come from one reference scenario, **8 galaxies with today's kernels**: the larger of G and LOO there, highest first. P0 ≥ 1.25, P1 ≥ 1.07.

A single scenario is used on purpose. Taking the maximum over scenarios lets a feature rank high on one column, and substitutes can both get credit that only one of them earns.

Complexity is a rough judgement of how much of the stack a feature touches:
* **low**: one op or the scheduler, plus validation;
* **med**: a new op or allocator, or changes across a few ops;
* **high**: cross-cutting changes to the runtime, KV layout, several kernels and the scheduler.

It is not a time estimate and has not been checked with the code owners.

| tier | feature | 4gx today | 4gx roofline | 8gx today (ranking) | 8gx roofline | complexity |
|---|---|---|---|---|---|---|
| P0 | KV offload tiers (host DRAM + 32 TB SSD/gx) | ×4.83 #1 / ×2.31 | ×9.91 #1 / ×2.60 | ×3.99 #1 / ×2.23 | ×5.51 #1 / ×2.79 | high |
| P0 | slot lanes + paged KV pool | ×1.54 #2 / ×1.90 | ×1.24 #3 / ×1.76 | ×1.22 #3 / ×2.22 | ×1.38 #3 / ×2.27 | high |
| P0 | async stage handoff | ×1.45 #4 / ×1.26 | ×1.37 #2 / ×1.30 | ×1.37 #2 / ×1.74 | ×1.72 #2 / ×2.29 | med |
| P0 | multi-request batching | ×1.17 #3 / ×1.47 | ×1.17 #4 / ×1.35 | ×1.44 #4 / ×1.63 | ×1.09 #4 / ×1.24 | high |
| P1 | variable chunk (a2a KV write) | ×1.10 #5 / ×1.09 | ×1.01 #7 / ×1.01 | ×1.14 #5 / ×1.17 | ×1.08 #7 / ×1.06 | high |
| P1 | variable-size lanes (arena) | ×0.91 #11 / ×0.91 | ×1.00 #9 / ×1.00 | ×1.02 #7 / ×1.08 | ×1.00 #10 / ×1.00 | med |
| P2 | fused multi-user attention | ×1.05 #8 / ×1.00 | ×1.00 #10 / ×1.00 | ×1.04 #8 / ×1.04 | ×1.00 #8 / ×1.01 | high |
| P2 | MSA SP-local indexer | ×1.04 #7 / ×1.03 | ×1.01 #8 / ×1.01 | ×1.03 #6 / ×1.03 | ×1.00 #9 / ×1.01 | high |
| P2 | index_k bf8 | ×1.00 #10 / ×1.00 | ×1.10 #6 / ×1.11 | ×1.01 #9 / ×1.00 | ×1.04 #6 / ×1.10 | low (PCC) |
| P2 | index_k stored once (not ×TP) | ×1.00 #9 / ×1.00 | ×1.17 #5 / ×1.13 | ×1.00 #10 / ×1.01 | ×1.16 #5 / ×1.11 | med |
| P2 | shortest-first run to completion | ×1.03 #6 / ×1.01 | ×0.99 #11 / ×0.99 | ×0.98 #11 / ×0.98 | ×1.00 #11 / ×1.00 | low |

Takeaways (study numbers are from the Oct 7 study unless marked Oct 1, the previous study with a 1 TB / 64 GB/s SSD tier in place of host DRAM + 32 TB SSD; the results of separate `tools/*.js` runs cited below, e.g. layout, batch-shape, SLO and fill diagnostics, were measured under older defaults, i.e. run to completion, double-buffered copies and 4 fixed lanes, with unaligned resume and, where a tier is on, the 1 TB SSD tier; they have not been re-run):
1. **On AgentX, KV capacity is the first wall; with the offload tiers it is gone with today's kernels, and tier bandwidth is the next wall with roofline kernels.** The best stacks reach 97–99% of their own ∞-cache goodput with today's kernels and 88–89% with roofline kernels (Oct 1, 1 TB SSD tier: 40–75%).
   * The ∞-cache column understates the roofline gap: the winners there moved to configs with a lower compute ceiling. With an infinite cache, the Oct 1 winners reach 209k at 4 gx (16×[4,2]) and 367k at 8 gx (budget 8k), against 129k and 289k for the new winners with the tiers.
   * The tiers and the pool are the top features in every scenario. Behind static slots the tiers alone give ×4.0–9.9; the pool then adds ×1.22–1.54 by sharing prefixes and freeing the slots' memory.
   * Tier bandwidth now matters more than capacity. On the best configs:
     * SSD 8 TB instead of 32 TB per galaxy changes nothing (≤0.1%), and PCIe at 181 GB/s instead of 63 (relaying through the x8 chips) gains nothing.
     * Half the SSD bandwidth costs 0–2% with today's kernels and 22–24% with roofline kernels (4 gx 129k → 100k, 8 gx 289k → 218k).
     * No host DRAM tier (SSD only) costs 0–1% with today's kernels and 16–17% with roofline kernels.
     * Oct 1, with the 1 TB tier, it was the other way round: 0.5 → 2 TB/galaxy moved 4-gx goodput 36.4k → 54.5k (today's kernels), and 64 → 16 GB/s cost at most 7%.
   * The lane table must be per stage: a global lane table (today's slot_id) drops the best configs 3.9× at 4 gx (60.7k → 15.5k) and 7.2× at 8 gx (150k → 20.8k) with today's kernels, and 2.0× at 8 gx with roofline kernels (289k → 143k). The 4-stage [8,4] winner at 4 gx with roofline kernels loses only 10%.
2. **Async stage handoff: ×1.26–2.29,** growing with pipeline depth and kernel speed (×1.45 when added / ×1.26 leave-one-out at 4 gx today, ×1.72 / ×2.29 at 8 gx with roofline kernels). The measured blocking send is 6–23 ms per chunk per stage.
3. **Batching: ×1.09–1.63,** most at 8 gx with today's kernels (×1.44 when added, ×1.63 leave-one-out). Batches in the best configs average 1.7–4.3 requests at the goodput point (4.3 at 8 gx today).
   * On today's 4-gx config + pool + 1 TB SSD tier (Oct 1 defaults) they averaged 1.33 requests; the average chunk was 7.0k of the 16k budget.
   * Most of the gain comes from one request taking several chunk-units at once (big cold prefills in fewer, larger chunks), not from mixing users.
   * **Dynamic batch size (`batchDynShape`, on by default).** A batch that is not full (e.g. a single request) runs at the tokens it holds, rounded up to whole chunks, as ops do without tracing. Off = padded to the full budget, as a traced build with one fixed shape must be; routed MoE ops still trim to the real tokens (`tools/batch_shape_ab.js`):
     * Today's 4-gx config + pool + host, budget 4k / 8k / 16k / 32k: sized 26.9k / 28.1k / 28.0k / 22.4k vs static 26.1k / 25.5k / 15.5k / 8.2k. No batching is 23.0k.
       * A static 16k or 32k shape collapses below no batching (71–85% padding): batches there hold only 1.1–1.2 requests.
     * Best stacks: static 16k costs 0–6% (4 gx today 45.2k → 42.3k; 8 gx 1%), and static 32k costs 1–21%.
       * These batches are fuller (2–3 requests), and cold prefills fill the budget.
     * A static shape near the typical fill (8k) is within 1% of dynamic sizes on the best stacks, but 9% behind on today's config (25.5k vs 28.1k).
   * With pool (4 lanes) but no SSD tier, batching adds only 4% (13.6k → 14.2k): the goodput point is set by cache misses (31–42% of prefilled tokens are re-prefill), not compute (`tools/cache_batch_ab.js`).
   * Paging vs pool: +10% without the SSD tier (15.0k vs 13.6k; paging frees the 4M lane tokens), +4% with the 1 TB tier (24.0k vs 23.0k). Pool/paging + 1 TB SSD tier come within 4–8% of the ∞-cache goodput of this compute config (25.1k), so most further gains must come from compute and TTFT. Sequential per-request attention is as good as fused, so the proposed plan (batch the MoE, attention per request) is the right one, and fused attention is not worth building. Best budget: 16k.
4. **Bounded dense gather is already done** (tt-metal #47539). Without it (the old whole-lane gather), the best 4-gx config with today's kernels (6 fixed 1M lanes) would drop 60.7k → 31.5k. Configs with arena lanes are unaffected.
   * The M3 comments that say the dense layers gather the whole cache shard (`prefill.py`, `tt_prefill_runtime.reconfigure_capacity`, README `PREFILL_MAX_SEQ_LEN`) are stale. Only the gather buffer is capacity-sized, which costs memory, not time.
5. **index_k stored once: ×1.11–1.17 with roofline kernels;** bf8 index_k adds another ×1.04–1.11. With today's kernels both are ×1.00–1.01: the tiers already hold the working set, while with roofline kernels smaller KV means less to read back from the tiers. (Oct 1, 1 TB tier: ×1.05–1.13 and ×1.05–1.08 everywhere.)
6. **Variable chunk / a2a KV write is worth 1–17%,** most with today's kernels (×1.10 / ×1.09 at 4 gx, ×1.14 / ×1.17 at 8 gx; Oct 1: 1–8%). (Before unaligned resume (#57636) joined the baseline it showed up to 20%: most of that was the chunk-rounding loss on resume.) The a2a KV write before the cache write is costed on every layer.
   * **A small fixed chunk plus batching gets nearly all of it** (`tools/layout_ab.js`, `results/layout_ab.json`, same stack and topology as the Oct 1 best grid config, 1 TB SSD tier).
     * The budget is always 16k. The best chunk is 128–256 (45.2k / 81.7k / 100.6k / 163.2k), which matches variable layout + fused attention within 1%.
     * Chunk 1024 is 0.2–2.4% behind that, and chunk 2048 is 1–3% behind 1024.
     * Batching itself is worth ×1.15–1.24 over the best unbatched fixed chunk.
     * Fixed layout here means each request takes whole C-token units, padded to C (8% padding at C=1024).
     * Chunk 1024 is extrapolated: the model was calibrated at 2048 and 5120.
   * **Batch size and compute:** per-token MoE cost at [4,2] (today's kernels) falls 12% from 4k to 8k, 6% from 8k to 16k and 3% from 16k to 32k. The expert matmuls cross from weight-bound to compute-bound at about 8k tokens per chunk. Goodput is 1–2.4% higher at 16k than at 8k; 4k is 6–10% worse; 32k is worse with today's kernels, because the longer period pushes p90 TTFT over the SLO.
   * **Smaller chunks (down to 32·SP = 128)** only cut padding once a request's chunk-units form one attention call. C=128 matches the variable layout and is 1–2% better than C=1024.
   * **One prefix gather per request per chunk (`kvDedup`) is essential for small chunks.** With one attention call and prefix gather per chunk-unit instead:
     * small chunks collapse, e.g. C=128 at 16k gives 2.2k instead of 100.5k at 8 gx today;
     * the best choice becomes C=2048, which is 2–14% below the de-duplicated best (8 gx today: 87.0k vs 100.6k).
     * Example: a cold 16k-token segment at 140k context pays 2.7 ms of gather per MoE layer as one call, but 41 ms as 16 × 1024 units.
   * **Prefetching the KV-prefix gathers (`prefetchKV`) is worth only 0.2–1.3%.**
     * In a synthetic 16k batch of 8 requests at 140k context it cuts the MoE layer 69 → 48 ms.
     * On AgentX at the goodput point, batches average 1.8–3 requests, so the gathers are small.
     * At 8 gx the bottleneck is the three single-dense-layer stages (93–95% utilisation vs 86% for MoE stages). Their full ring-joint attention is compute-bound, so prefetch barely helps them.
     * The next lever there is dense attention itself: ring-joint compute runs at 33% efficiency, or the dense layers could get more chips.
   * Variable layout + fused attention is the model's version of **ragged (varlen) attention**: packed segments padded only to 32·SP, one attention launch per chunk, and each segment attends only to its own context. It assumes today's per-op efficiencies, not a faster kernel.
   * **The same questions with an infinite cache** (compute-bound view; `tools/layout_ab.js --inf` → `results/layout_ab_inf.{json,txt}`, `tools/budget_ext_inf.js`, `tools/fused_fixed_inf.js`). The findings above use the real cache (pool + 1 TB/gx SSD tier), where the cache limits concurrency.
     * **Best fixed setup:** chunk 128 everywhere, and chunk size now matters more:
       * chunk 1024 is 1–6% behind and chunk 2048 is 7–13% behind;
       * budget 16k with today's kernels (8k is −7–8%; ≥32k collapses on TTFT);
       * budget 64k with roofline kernels (still rising: 32k → 48k → 64k = 216.5 → 223.5 → 227.0k at 4 gx, 426.9 → 439.5 → 444.7k at 8 gx).
       * Goodput: 60.9k / 227.0k / 135.1k / 444.7k (4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline).
     * **Variable chunk size adds nothing.** Var layout + per-request attention is 0.3–2.6% *below* fixed chunk 128.
     * **Ragged (fused) attention is what helps, and it works on the fixed chunk-128 layout too** (fixed + fused ≥ var + fused everywhere):
       * +8.8% at 8 gx today (135.1 → 147.0k), where the single-dense-layer stages are the bottleneck and a fused call pays the fitted 2.75 ms per-call dense-attention cost once per chunk instead of once per request;
       * +1.8% at 4 gx today, +0.7–1.7% with roofline kernels.
       * This relies on that per-call cost being launch/setup-like, which is not verified.
     * **Prefetching the KV-prefix gathers: +2–6%** on top of fixed + fused (4 gx today 62.0 → 65.6k, 8 gx today 147.0 → 150.9k, roofline +1.9–2.5%), vs 0.2–1.3% with the real cache.
     * **One prefix gather per request per chunk (`kvDedup`) matters even more.** Without it, the best setup is chunk 2048 and is 13–24% lower.
     * **Peak throughput (any TTFT), infinite cache** (`tools/layout_ab.js --inf --slo none --budgets …,65536`, `results/layout_ab_inf_peak.{json,txt}`). Same conclusions:
       * Best fixed setup: chunk 128 with budget 64k (the largest tried, still rising 1.5–3.5% from 32k). Goodput: 67.8k / 228.8k / 152.2k / 444.8k.
       * Relative to that: chunk 1024 is 3–6% lower, chunk 2048 is 9–13% lower.
       * Variable chunk alone: −0.4 to −0.9%.
       * Ragged (fused) attention on the fixed layout: +0.3–1.8%, and +7.0% at 8 gx today.
       * Prefetch: +1.7–3.3%.
       * Without `kvDedup`: 14–29% lower.
       * **Where ragged attention and prefetch act** (`tools/attn_diag.js`, `results/attn_diag.txt`):
         * **Sparse (MSA) layers:** each request gathers its own K/V + index prefix, which takes longer than its indexer + sparse attention (2.3 vs 1.3 ms per request at 140k with today's kernels, 0.4 vs 0.04 with roofline). A fused call cannot share those gathers, so ragged attention saves only 0–3% of an MSA layer (latency floors, core fill).
         * **Dense layers:** ragged attention saves 11–31% at 4–10 requests per batch. That is almost entirely the fitted fixed cost of about 2.75 ms per ring-joint call, paid once per chunk instead of once per request. It is not verified that this cost is per call: time a dense layer with 1 vs several segments.
         * The gain grows with requests per batch, not with smaller chunks (chunk 128 vs 1024 changes it by ≤1.3 points).
         * **Prefetch** cuts an MSA layer by 3–5% and a dense layer by 0–3%. That is 3% with today's kernels, where the dense ring gather exceeds its compute, and 0% with roofline kernels, where it doesn't.
         * **Bottleneck at the peak:** with today's kernels at 8 gx the three single-dense-layer stages are at 100% (MoE stages 94%). Prefetch still gives +3% there only because it hides part of the dense gather. In the other scenarios the MoE stages are the bottleneck (dense stages 41–75%).
       * With today's kernels the peak is 11–13% above goodput at 10 s (bigger budgets, 10–13 requests per batch). With roofline kernels goodput is already the peak.
7. **Topology** (best of the grid per topology; seeds move results by up to 5%):
   * [4,2] is best in three scenarios, by 2–12% over the next topology; it needs KV heads sharded 2 per chip. At 4 gx with roofline kernels 4×[8,4] stages win by 3% (129.0k vs 125.3k for 16×[4,2]).
   * **[4,4] torus stages:** with ring collectives they now lose everywhere: 55.4k vs 60.7k at 4 gx today, 103k vs 150k at 8 gx today, 259k vs 289k at 8 gx with roofline kernels (Oct 1 they tied at 4 gx today and at 8 gx roofline).
     * Per chip, a [4,4] MoE layer is still about 20% more expensive than a [4,2] one. The TP=4 collectives cost more, and a 16-chip stage pays the same fixed per-op latency as an 8-chip one.
     * Rings alone give [4,4] +5–12% (`tools/torus_ab.js`).
   * Rings on every 4-long axis ([4,2]'s SP axis, [2,4]'s TP axis) would add another 16–19% to [4,2] with today's kernels: 4 gx 60.7k → 72.0k, 8 gx 150k → 174k. With roofline kernels they add nothing.
   * 8 gx: one 32-stage pipeline beats 2×16 (134k vs 102k today, 284k vs 242k roofline, both [2,4]). [8,4] stages lose at 8 gx.
8. **Faster decode raises prefill goodput with roofline kernels** (90 → 360 tok/s: 112k → 137k at 4 gx, 243k → 296k at 8 gx), because it shrinks the live KV working set per unit of load. With today's kernels it moves goodput within seed noise (4 gx 63.5k → 60.1k, 8 gx 152k → 149k). Oct 1, with the 1 TB tier: 42.7k → 46.7k at 4 gx today.
9. **Pool copies:** double-buffered copies are within 0.5% of sequential; triple buffering costs 5% at 4 gx and 13% at 8 gx with today's kernels (its lanes take memory), and nothing with roofline kernels. 4 fixed 1M lanes are 1–4% behind an arena.
10. **The SLO is not what limits batch fill; KV capacity is** (`tools/slo_ab.js`, `results/slo_ab.txt`; Oct 1 defaults with the 1 TB SSD tier, except where marked Oct 7).
    * Requests per batch at p90 ≤ 10 s → no SLO: 1.33 → 1.98 (today's 4-gx config + pool + host + batch) and 1.8–2.9 → 1.8–3.1 (best stacks).
    * Goodput gains only 0–1% on the best stacks (+2% on today's config). Oct 7: the SLO-free peak of the best configs is 0–3% above their goodput. Beyond that point more sessions stop fitting in the KV cache, so the throughput peak sits at about the same concurrency as the 10 s point.
    * With an infinite cache, batches fill: 2.8–4.3 requests per batch (14–16k of the 16k budget) at 10 s, 3.8–4.5 with no SLO, and 96–99% of the bottleneck stage busy.
    * **Why the real cache fills batches less** (`tools/fill_diag.js`, `results/fill_diag.txt`: same stack, same concurrency, KV residency varied):
      * Pool lanes vs full paging and tier bandwidth make no difference. **Tier capacity is the whole gap**: at 8 TB/galaxy instead of 1 TB, every metric matches the infinite cache.
      * Past about 480 streams at 4 gx (1136 at 8 gx), the live AgentX working set no longer fits device + 1 TB/gx host, and evicted prefixes come back as re-prefill.
        * At the infinite cache's peak concurrency, re-prefill is ~60% of prefilled tokens (vs 3–5%), hit rate drops from 96% to 90%, and useful throughput halves.
        * So the SLO point sits at lower concurrency. Fewer requests are in prefill at once, and batches hold fewer.
      * At equal concurrency the real cache's batches are, if anything, fuller (4 gx at 480: 2.01 vs 1.44 requests), because re-prefill makes each request longer.
      * That is why the tier behind host DRAM is SSD: 8 TB/galaxy of host DRAM would exceed the AgentX host-DRAM cap, while NVMe capacity is cheap. The Oct 7 study bears this out: with 32 TB of SSD per galaxy, the best configs with today's kernels come within 1–3% of the infinite cache. Smaller KV (index_k fixes, bf8) also helps.
    * A 5 s SLO costs 20–40% with today's kernels, and 0–2% with roofline kernels.
    * The page has an SLO slider, whose right end means no SLO. A no-SLO run bisects the throughput peak.
      * Any sweep whose throughput peaks before the SLO crossing also bisects the peak (`lib/sweep.js` step 4). Before that fix, a re-run with a looser SLO could report up to 3% *lower* goodput (4 gx roofline: 81.4k at 10 s, 78.9k at 30 s), because its bisection moved past the peak. Now it is within 0.2% across 5 s … no SLO.
      * `results/study.json` (Oct 7) includes the fix.
11. **The cliff is steep.** Goodput can change by 10% between neighbouring concurrency points, so the sweep bisects the SLO crossing.

## Assumptions to revisit

* Offload tiers: capacities and bandwidths come from the Galaxy Blackhole documentation and the drive datasheet, but the host DRAM reserves (OS, runtime, KV staging) are estimates, and transfers are ideal page DMAs. The PCIe figure assumes each chip moves its own KV over its own link; measure the sustained rate.
* Pool copies are page-list gathers at 50% DRAM efficiency. Arena fragmentation is not modelled.
* Decode is a fixed per-request rate, whatever the number of sessions decoding, and its capacity is a slot count
  (`decodeSlots`), not KV bytes per context length (today's decode slots hold 64k positions; AgentX contexts average
  about 130k). KV migration to decode is not modelled.
* Meshes without a profile ([4,4], [1,4], …) are extrapolated. TP=2 stages use the [4,2] single-stage profile.
* Ring-collective speed-ups on the torus are textbook link-load ratios, not measured; profile a [4,4] stage with ring CCLs to pin them down.
* Batched chunks larger than 5120 are extrapolated from the roofline scaling of each op.

## Extending (notes for the next agent)

* **New feature.** Add a knob to `DEFAULTS` in `sim_core.js`, use it in `roofTok` / `roofSeg` / `layerMs` (cost), `makePlan` (memory), or the scheduler (`formChunk`, `tryStart`). Then add a `FEATURES` entry in `study.js`, a control in `artifact/template.html` (`FIELDS`) and a complexity line in `build_artifact.js` (`SCOPE`).
* **New hardware data.** Rerun `collect_calib.py`. It segments the timing CSVs per cell with `results_16stage.jsonl`. Then run `validate.js` and check the error summary before trusting a study.
* **New corpus.** Rerun `prep_traffic.py`. hash_ids must stay prefix-chained and topologically increasing (checked on 19.6M block pairs).
* `lib/pool.js` `summarize` and the copy in `artifact/template.html` must stay identical.
