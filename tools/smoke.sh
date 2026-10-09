#!/bin/bash
# Smoke test every feature path at one concurrency. Usage: JOB=<slurm job> tools/smoke.sh [C]
cd "$(dirname "$0")/.." || exit 1
NODE=${NODE:-node}
C=${1:-256}
run() { echo "## $*"; ./on_node.sh "$NODE" run.js --preset today-C --conc "$C" "$@" 2>&1 | grep -v '^plan'; }
run
run --set boundedDense=true
run --set boundedDense=true --set cache=paging
run --set boundedDense=true --set cache=inf
run --set boundedDense=true --set cache=pool --set lanes=3
run --set boundedDense=true --set cache=pool --set laneArena=true --set arenaTokens=3000000
run --set boundedDense=true --set cache=pool --set lanes=3 --set hostTier=true
run --set boundedDense=true --set cache=inf --set unaligned=true
run --set boundedDense=true --set cache=inf --set reqPad=tile
run --set boundedDense=true --set cache=inf --set reqPad=tile --set batch=true --set budget=16384
run --set boundedDense=true --set cache=inf --set reqPad=chunk --set batch=true --set budget=8192
run --set boundedDense=true --set cache=inf --set asyncHandoff=true
run --set boundedDense=true --set cache=inf --set opEff=1
run --set boundedDense=true --set cache=inf --set opEff=1 --set reqPad=tile --set batch=true --set budget=16384 --set asyncHandoff=true
run --set boundedDense=true --set cache=inf --set policy=srpt
run --set policy=rr --set cache=pool --set lanesOverride=true --set lanes=4 --set batch=true --set chunk=1024 --set budget=8192
run --set policy=rr --set cache=pool --set copyMode=double --set hostTier=true
run --set policy=rr --set cache=paging
run --set cache=pool --set hostTier=true --set decodeStages=62 --set decodeSlots=75 --set decodeHostTier=true
run --set cache=pool --set decodeStages=62 --set decodeSlots=75 --set decodeCache=hybrid --set decodeLanes=62 --set decodeHostTier=true
run --set cache=pool --set decodeStages=62 --set decodeSlots=75 --set decodeCache=hybrid --set decodeLaneScope=stage --set decodeLanes=1
run --set cache=pool --set decodeStages=62 --set decodeSlots=75 --set decodeCache=paging --set decodeHostTier=true
run --set cache=pool --set hostTier=true --set decodeStages=62 --set decodeSlots=86 --set decodeHostTier=true --set decodeBackpressure=queue
