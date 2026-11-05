#!/bin/bash
set -e
cd /disk1/query-evidance
export CE_DATASET_URL=http://10.52.101.89:8811
declare -A M=( [musique200]=musique [wiki2_200]=wiki2 [hotpot200]=hotpot [squad_200]=squad [trivia_200]=trivia )
for tag in "$@"; do
  export CE_DATASET=${M[$tag]}
  echo "=== $tag with ${M[$tag]} ==="
  python3 -m src.g8_equal_ce --n 200 --data ${tag}.jsonl > logs/ce_${tag}_base.log 2>&1
  python3 -m src.e13_exec --n 200 --workers 10 --modes select --data ${tag}.jsonl > logs/ce_${tag}_rbg.log 2>&1
  echo "done $tag"
done
