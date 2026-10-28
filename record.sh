#!/usr/bin/env bash
# Records open interest, funding and order flow for 20 Phemex perps into ./data, restarting after any crash.
cd "$(dirname "$0")"
while true; do
  python3 -m scalper record-all
  echo "recorder stopped; restarting in 10 s (Ctrl+C to quit)"
  sleep 10
done
