#!/usr/bin/env bash
# Paper trades the two three-move setups on 20 Phemex perps, restarting after any crash. No orders are ever sent.
cd "$(dirname "$0")"
while true; do
  python -m scalper paper-3move
  echo "paper bot stopped; restarting in 60 s (Ctrl+C to quit)"
  sleep 60
done
