#!/bin/bash
# Round-based extraction runner: 405B -> 70B -> 8B, raw then template.
# Serial within a round (one NDIF request in flight). A failed run does NOT
# block the others: it is retried next round, chunks already on HF are skipped.
cd /root/plot-pipeline
declare -A ok
for round in $(seq 1 20); do
  pending=0
  for m in 405b 70b 8b; do
    for regime in raw template; do
      key="${m}_${regime}"
      [ "${ok[$key]}" = "1" ] && continue
      flag=""
      [ "$regime" = "raw" ] && flag="--no-chat-template"
      echo "=== $(date -u +%H:%M:%S) round $round starting $m $regime ===" | tee -a logs/sequence.log
      .venv/bin/python scripts/ndif_extract_devbunova.py --model $m $flag >> "logs/${m}_${regime}.log" 2>&1
      rc=$?
      echo "=== $(date -u +%H:%M:%S) round $round finished $m $regime rc=$rc ===" | tee -a logs/sequence.log
      if [ $rc -eq 0 ]; then ok[$key]=1; else pending=1; fi
    done
  done
  if [ $pending -eq 0 ]; then
    echo "=== ALL SIX RUNS DONE $(date -u) ===" | tee -a logs/sequence.log
    exit 0
  fi
  echo "=== round $round had failures; cooling down 600s ===" | tee -a logs/sequence.log
  sleep 600
done
echo "=== GIVING UP after 20 rounds $(date -u) ===" | tee -a logs/sequence.log
exit 1
