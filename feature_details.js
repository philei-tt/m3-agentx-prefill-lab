// Per-feature explanations shown in the artifact's Roadmap table (click a feature to expand).
// what: what the feature is and where it sits today; why: mechanism + numbers; todo: rough implementation steps;
// notes: interactions and caveats. Keys match study.js FEATURES.
'use strict';
module.exports = {
  pool: {
    what: 'Replace the static 1M-token slots with a few contiguous <b>lanes</b> per stage plus a shared <b>paged pool</b>. Lanes are the KV buffers the existing attention kernels run on. The pool is the rest of KV memory, split into pages keyed by the hash of their content (64-token blocks), shared by every conversation with the same prefix and evicted least-recently-used. Before a request runs, its cached prefix is copied from pool pages into a lane; afterwards its new tokens are copied back to the pool.',
    why: 'Today a conversation owns a whole 1M slot even when its context is 50k tokens, so 4 galaxies keep only 20 conversations warm. AgentX needs hundreds of live sessions to fill the pipeline, so almost every request misses: 0.3% hit rate at 256 lanes in the simulation. A pool stores only the tokens that exist: the same 21M tokens hold about 150 median (140k) contexts instead of 20. It shares identical prefixes (sub-agents start from the main agent\'s system prompt and tools) and evicts from the tail, so a conversation keeps its head. The copies are cheap: a median hit is about 65-107 MB per chip, about 0.5 ms of DRAM time, overlapped with compute.',
    todo: [
      'Page allocator and page table per stage in device DRAM; prefix index from block hash to page on the host.',
      'A page-list gather/scatter op that copies many pages into or out of a lane in one launch (per-page host copies would be dispatch-bound).',
      'A per-stage lane table: each stage allocates a lane only while it works on the request (today slot_id is global across stages). This is required once batching is on: a global lane table drops the best 4-galaxy config from 45k to about 14k useful tok/s (8 galaxies: 102k to 28k).',
      'Pool copies double-buffered: copy-in of the next batch and copy-out of the previous one overlap compute. The peak lane memory is about 2x the computing lanes, briefly; sequential copies (1x) cost about 1% on the best configs.',
      'Scheduler: LRU eviction, pinning of in-flight prefixes, copy-in of the next request while the current chunk computes, copy-out after the last chunk.',
      'KV migration to decode reads from lane or pool pages instead of a fixed slot.',
    ],
    notes: 'Prerequisite for the SSD KV tier. The lane count limits how many requests can share a batched chunk.',
  },
  host: {
    what: 'A second KV tier on NVMe SSDs in the galaxy hosts, behind the device pool or the static slots. With the pool, pages the device evicts are written to SSD instead of dropped; with static slots, a reclaimed slot is written to SSD and read back when its conversation returns (one copy per conversation, no prefix sharing). When a request hits an SSD-resident prefix, those pages are read back into its lane while the request waits in the queue.',
    why: 'M3 KV is about 127 KB per token today (73 KB with the index-cache fixes). 4 galaxies hold about 21M tokens on device, but the live working set of AgentX near the throughput limit is several times larger, so the device pool alone still evicts conversations that come back minutes later. Each TB of SSD per galaxy adds about 32M tokens (56M with the index fixes). A hit costs a read: a 150k-token prefix is about 19 GB, about 75 ms at 4 × 64 GB/s or about 0.3 s at 4 × 16 GB/s, done while the request is queued. In the simulation capacity matters a lot (0.5 → 2 TB per galaxy: 36.8k → 54.5k goodput at 4 galaxies with today\'s kernels; 8 TB matches an infinite cache) and bandwidth much less (64 → 16 GB/s per galaxy costs at most 13%).',
    todo: [
      'SSD page store with its own LRU and an index from block hash to file offset.',
      'Asynchronous device→SSD write-back on eviction and SSD→device read before admission (GPUDirect-style or a pinned host bounce buffer).',
      'Scheduler: start the read when the request arrives; a fetching request must not block the pipeline.',
      'Measure the real drive capacity and sustained read/write bandwidth per galaxy host; the model uses one symmetric bandwidth (placeholder 1 TB, 64 GB/s per galaxy).',
      'Watch write endurance: evictions are written continuously under load.',
    ],
    notes: 'Works behind the pool or static slots. Capacity is what matters; check how the AgentX rules treat SSD-backed KV (they cap host DRAM per system).',
  },
  idxdedup: {
    what: 'The index-key cache used by the MSA indexer (one 128-wide key per token per layer, bf16) is stored on all 4 TP columns. That is 1024 of the 2112 bytes per token per layer: 48% of all KV memory. Store it once per SP row and gather it inside the indexer. The dense layers also allocate a zero-filled index cache, which can be dropped.',
    why: 'KV per token per layer drops from 2112 to 1344 bytes (−36%), so the same memory holds 57% more tokens. KV capacity is what limits goodput on this traffic, so the gain is ×1.06-1.16. It used to look larger (×1.2-1.3), but part of that was second-tier capacity, and SSD copies now store index_k once anyway.',
    todo: [
      'Allocate index_k sharded over TP (by token blocks) or on one TP column, instead of replicated.',
      'In the indexer, all-gather index keys over TP before scoring, or score per TP shard and merge top-k. The extra traffic is small next to the K/V gather.',
      'Update the KV migration table and the decode-side layout.',
      'PCC.',
    ],
    notes: 'Stacks with index_k bf8. Its value grows with more lanes (more concurrent contexts).',
  },
  idxbf8: {
    what: 'Store the index-key cache in bf8 (1.0625 B per value) instead of bf16 (2 B).',
    why: 'KV per token per layer drops from 2112 to 1632 bytes (from 1344 to 1224 with de-replication): about 10-30% more capacity, for ×1.05-1.11 goodput.',
    todo: [
      'bf8 is already the runner\'s default; the measured runs opt into bf16 with M3_INDEX_CACHE_BF16=1.',
      'Validate that top-k block selection and end-to-end PCC hold at long contexts with bf8 index keys.',
    ],
    notes: 'Mostly an accuracy sign-off, not implementation work.',
  },
  async: {
    what: 'Today each stage receives a chunk, computes it, then sends the activation to the next stage (about 63 MB for a 5120-token chunk), and the send blocks the stage. Fitted from runs A/B/C: about 23 ms per chunk at 5120 (8 ms at 2048) of blocking send, plus 14-19 ms of hop latency. With async handoff the send overlaps with computing the next chunk.',
    why: 'The blocking send adds 10-35% to the pipeline period. Its share grows with more stages (fewer layers per stage) and with faster kernels: ×1.12-1.20 at 4 galaxies with today\'s kernels, ×1.45-1.62 at 8 galaxies with roofline kernels. It also cuts TTFT, since 32 hops × 15 ms is about 0.5 s.',
    todo: [
      'Double-buffered D2D send/receive: post the send and start the next chunk immediately; the receiver pre-posts buffers.',
      'Make sure the transfer uses enough links (the activation is spread over the stage\'s chips).',
      'Re-measure with PREFILL_SYNC_PER_CHUNK=0: the per-chunk sync used for timing in the benchmark may be part of the measured blocking.',
    ],
    notes: 'Independent of the other features. Its value grows as the kernels get faster.',
  },
  batch: {
    what: 'Put several requests\' new tokens into one chunk, up to a token budget (8-16k works best). Projections, norms and the whole MoE run on the concatenated tokens; attention runs per request (sequential).',
    why: 'Most AgentX requests are small (median 1600 new tokens), so most chunks are small. An MoE layer has a large per-chunk cost that does not depend on tokens: expert weights (16 experts × 32 MB per chip per layer at [2,4]) and about 10 collectives per layer with ~40 µs latency each. More tokens per chunk amortise it. At chunk 2048 each expert sees only about 64 tokens (T×4/128), far below the 260-460 tokens per expert where the matmuls become compute-bound. Measured: ×1.14-1.29 when added, ×1.17-1.33 when removed from the full stack, at slightly higher TTFT. With sequential attention it does not matter which requests share a chunk.',
    todo: [
      'Chunk metadata per segment: lane/slot, start position, length.',
      'An attention loop over segments; each segment\'s KV is read and written in its own lane.',
      'MoE dispatch/combine buffers and activations sized for the budget instead of the chunk.',
      'Traces per chunk size (or a few buckets, e.g. 4k/8k/16k), so a partly filled batch is not padded to the budget. With one static 16k shape the batches here (1.2-3 requests) are 40-65% padding: -6% on the best stacks, and below no batching on today\'s config.',
      'One attention call per request per chunk, so a request\'s several chunk-units share one KV-prefix gather.',
      'A packing scheduler: fill the budget, split long requests across chunks, keep segments of one request in order.',
      'Needs the variable layout for token packing; with the fixed layout only whole chunks can be packed.',
    ],
    notes: 'Each request in a chunk needs a lane on the stage, so batching wants more (or variable-size) lanes. Fused multi-user attention adds nothing on top.',
  },
  var: {
    what: 'Today the chunk size is fixed per launch. KV is laid out block-cyclically over SP with period chunk/SP, RoPE is built with the same layout, the MoE dispatch buffers are sized by the chunk, and the MSA cache read requires cached_len % chunk == 0. Variable layout allows any segment length (a multiple of 32·SP tokens), and each layer\'s new K/V rows are routed to the SP device that owns them with a small all-to-all.',
    why: 'It removes padding: 15% of processed tokens at chunk 2048, and about half at 5120, on the AgentX mix. It also removes alignment loss on resume. It enables token-packed batching. The extra all-to-all is about 0.1 ms per layer.',
    todo: [
      'Absolute-position RoPE, independent of chunk size.',
      'An all-to-all KV write op, or a fixed fine-grained block-cyclic cache layout (e.g. 128-token blocks) written through a scatter.',
      'MSA cache read and ring attention without chunk-aligned starts.',
      'MoE buffers sized for the maximum budget; compile/trace buckets for variable token counts.',
    ],
    notes: 'Includes unaligned resume. Most of its value is realised together with batching. A small fixed chunk plus batching gets nearly all of it, as long as a request\'s chunk-units in one batch form one attention call (its prefix is gathered once). Fixed chunk 128-256 with a 16k budget matches variable layout + fused ("ragged") attention within 1% in every scenario; chunk 1024 is 0.2-2.4% behind (tools/layout_ab.js). Chunks below 2048 are extrapolated (calibrated at 2048 and 5120).',
  },
  unaligned: {
    what: 'Resume a conversation at any 32-token boundary instead of rounding the cached prefix down to a chunk multiple. PR #57636, in review.',
    why: 'With chunk 2048 the rounding throws away 1024 cached tokens per request on average, which is a large fraction of a typical 1600-token turn. up to ×1.12 when added before the variable layout.',
    todo: ['Land PR #57636 and run the two-turn 60-layer PCC check.'],
    notes: 'The variable layout includes it, which is why it shows ×1.00 in the full stack.',
  },
  arena: {
    what: 'Instead of fixed 1M lanes, give each request a lane of its actual length from one contiguous arena per stage (2-8M tokens).',
    why: 'Frees the memory that fixed lanes reserve (4 × 1M tokens per stage) for the pool, and lets many small requests be in flight at once for batching.',
    todo: [
      'Kernels take a base offset + length instead of a slot index.',
      'An allocator with fragmentation handling (compaction, or size classes).',
    ],
    notes: 'Its value depends on how many lanes batching needs; see the "4 fixed 1M lanes" sensitivity row.',
  },
  msa: {
    what: 'Today every MSA layer all-gathers the whole cached K/V and index keys over SP for each request (ag_kv / ag_index_k). Instead, each SP rank scores its own index keys, a small top-k merge picks the 16 blocks per query, and only those blocks are fetched.',
    why: 'The gather grows with context: at [2,4] about 0.007 ms per 1k cached tokens per layer, so about 1 ms per layer at 140k, against an 18 ms layer today. The gain is ×1.01-1.02 in the full stack at SP=2; one early greedy step shows ×1.10, but that is a configuration still dominated by cache thrashing; it would matter more with larger SP or much faster kernels.',
    todo: [
      'A distributed top-k merge.',
      'A block-fetch op for the selected K/V blocks.',
      'Changes to the indexer_score_msa / sparse_sdpa_msa inputs.',
    ],
    notes: 'Not worth it at [2,4] stages.',
  },
  fused: {
    what: 'One attention kernel over all requests in a batched chunk, instead of one attention call per request.',
    why: 'It could save launches and fill cores better for short segments. The model includes both effects (per-op latency and a 110-core wave-quantisation penalty for short segments). It finds ≤2%, because batched segments average about 1.6k tokens, which already fill the cores.',
    todo: ['New multi-user MSA and ring-joint kernels.'],
    notes: 'Not recommended: sequential per-request attention is as good. That confirms the "batch the MoE, attention per request" plan.',
  },
  srpt: {
    what: 'Replace round robin (the base, as tt-d-gen) with run to completion, the queue ordered by new tokens (shortest first) with 30 s aging so long requests are not starved.',
    why: 'Short requests stop waiting behind 50k-token prefills, so p90 TTFT drops and more load fits under the SLO: ×1.00-1.08.',
    todo: ['Scheduler-only change.'],
    notes: 'Cheap; its gain grows near saturation.',
  },
};
