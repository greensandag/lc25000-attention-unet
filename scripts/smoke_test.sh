#!/usr/bin/env bash
# End-to-end check on a tiny synthetic dataset with a tiny model (CPU, ~5 minutes).
# The numbers it produces are meaningless; it only checks that every stage runs.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=${1:-/tmp/lcseg_smoke}
rm -rf "$TMP"
python scripts/make_synthetic_lc25000.py --out "$TMP/data" --sources 5 --copies 6
C=tests/smoke_config.json
python -m lcseg prepare   --lc-root "$TMP/data" --out "$TMP/out" --config $C
python -m lcseg group     --out "$TMP/out" --config $C
python -m lcseg generator --out "$TMP/out" --config $C
python -m lcseg launch    --out "$TMP/out" --config $C
python -m lcseg bench     --out "$TMP/out" --config $C
python -m lcseg report    --out "$TMP/out" --config $C
echo "OK: see $TMP/out/report/results.md"
