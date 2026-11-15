#!/bin/bash
# Run every comparison with the authors' implementations. Correctness only: no timings.
#
#   bash checks/run_all.sh llada   # LLaDA-8B-Instruct cases, then d2Cache
#   bash checks/run_all.sh sdar    # block cache on SDAR (needs flash-attn and a C compiler)
#   bash checks/run_all.sh v2      # block and sub-block cache on Fast-dLLM v2
#
# UPSTREAM holds the reference checkouts at the revisions pinned on the method pages,
# OUT receives one JSON file per case. PY, PY_RDLLM and PY_SDAR are the interpreters of
# the environments described in docs/checks.md.
set -u
UPSTREAM=${UPSTREAM:-upstream}
OUT=${OUT:-checks/results}
PY=${PY:-python}
PY_RDLLM=${PY_RDLLM:-$PY}
PY_SDAR=${PY_SDAR:-$PY}
mkdir -p "$OUT"
quiet='warning\|Loading checkpoint\|generation flags\|trust_remote_code\|store_kv\|deprecated\|^\s*$'

run() {
  name=$1
  shift
  echo "=== $name"
  $PY checks/compare_upstream.py "$@" --out "$OUT/$name.json" 2>&1 | grep -v -i "$quiet" | tail -5
}

if [ "${1:-llada}" = sdar ]; then
  for dtype in float32 bfloat16; do
    echo "=== block-cache-sdar-$dtype"
    $PY_SDAR checks/compare_sdar.py --root "$UPSTREAM/sdar" --dtype $dtype \
      --out "$OUT/block-cache-sdar-$dtype.json" 2>&1 | grep -v -i "$quiet" | tail -5
  done
  echo "=== all done"
  exit 0
fi

if [ "${1:-llada}" = v2 ]; then
  for dtype in float32 bfloat16; do
    echo "=== fast-dllm-v2-$dtype"
    $PY checks/compare_fast_dllm_v2.py --dtype $dtype \
      --out "$OUT/fast-dllm-v2-$dtype.json" 2>&1 | grep -v -i "$quiet" | tail -5
  done
  echo "=== all done"
  exit 0
fi

run adapter --case adapter
run fast-dllm-none --case fast-dllm-none --root "$UPSTREAM/fast-dllm"
run fast-dllm-prefix --case fast-dllm-prefix --root "$UPSTREAM/fast-dllm"
run fast-dllm-prefix-th --case fast-dllm-prefix --root "$UPSTREAM/fast-dllm" --threshold 0.9
run fast-dllm-dual --case fast-dllm-dual --root "$UPSTREAM/fast-dllm"
run fast-dllm-dual-th --case fast-dllm-dual --root "$UPSTREAM/fast-dllm" --threshold 0.9
run dkv-cache-decode --case dkv-cache-decode --root "$UPSTREAM/dkv-cache" --refresh-interval 8
run dkv-cache-decode-r4 --case dkv-cache-decode --root "$UPSTREAM/dkv-cache" --refresh-interval 4
run dkv-cache-greedy --case dkv-cache-greedy --root "$UPSTREAM/dkv-cache" --refresh-interval 2 --greedy-window 4
run dllm-cache --case dllm-cache --root "$UPSTREAM/dllm-cache"
run dllm-cache-k5-3 --case dllm-cache --root "$UPSTREAM/dllm-cache" --prompt-interval 5 --response-interval 3
run dllm-cache-k5-3-r0 --case dllm-cache --root "$UPSTREAM/dllm-cache" --prompt-interval 5 --response-interval 3 --transfer-ratio 0.0
run elastic-cache --case elastic-cache --root "$UPSTREAM/elastic-cache"
run dpad-none --case dpad-none --root "$UPSTREAM/dpad" --window 128
run dpad-prefix --case dpad-prefix --root "$UPSTREAM/dpad" --window 128
run dpad-prefix-th --case dpad-prefix --root "$UPSTREAM/dpad" --window 128 --threshold 0.9
run dpad-dual --case dpad-dual --root "$UPSTREAM/dpad" --window 128
run sparse-dllm --case sparse-dllm --root "$UPSTREAM/sparse-dllm"
for dtype in float32 bfloat16; do
  echo "=== d2cache-$dtype"
  $PY_RDLLM checks/compare_d2cache.py --root "$UPSTREAM/d2cache" --dtype $dtype \
    --out "$OUT/d2cache-$dtype.json" 2>&1 | grep -v -i "$quiet" | tail -5
done
echo "=== all done"
