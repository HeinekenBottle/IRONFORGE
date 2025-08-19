#!/usr/bin/env bash
set -euo pipefail

SYM="${1:-NQ}"
TF="${2:-5}"
WIN="${3:-512}"
DATE_TAG="$(date +%F)"

OUT_BASE="runs/${DATE_TAG}"
RUN_A="${OUT_BASE}/${SYM}_${TF}m"        # HTF OFF (45D)
RUN_B="${OUT_BASE}/${SYM}_${TF}m_htf"    # HTF ON  (51D)

echo "== IRONFORGE RC Smoke :: ${SYM} ${TF}m on ${DATE_TAG} =="

# Optional prep (dry-run ok if shards already exist)
python -m ironforge.sdk.cli prep-shards --symbol "$SYM" --tf "M${TF}" --dry-run || true
python -m ironforge.sdk.cli prep-shards --symbol "$SYM" --tf "M${TF}" --htf-context --dry-run || true

# --- HTF OFF (45D)
python -m ironforge.sdk.cli discover-temporal --symbol "$SYM" --tf "$TF" --window_bars "$WIN" --out "$RUN_A"
python -m ironforge.sdk.cli score-session    --run    "${RUN_A}*"
python -m ironforge.sdk.cli validate-run     --run    "${RUN_A}*" --kfold 5 --embargo 20
python -m ironforge.sdk.cli report-minimal   --run    "${RUN_A}*" --out "${OUT_BASE}/report_45d/"

# --- HTF ON (51D)
python -m ironforge.sdk.cli discover-temporal --symbol "$SYM" --tf "$TF" --window_bars "$WIN" --htf-context --out "$RUN_B"
python -m ironforge.sdk.cli score-session     --run    "${RUN_B}*"
python -m ironforge.sdk.cli validate-run      --run    "${RUN_B}*" --kfold 5 --embargo 20
python -m ironforge.sdk.cli report-minimal    --run    "${RUN_B}*" --out "${OUT_BASE}/report_51d/"

# Status
python -m ironforge.sdk.cli status --runs runs || true

# Contract validations (both runs)
python scripts/validate_contracts.py "${RUN_A}" --expect-node-dims 45 --name "${SYM}_${TF}m_45d"
python scripts/validate_contracts.py "${RUN_B}" --expect-node-dims 51 --name "${SYM}_${TF}m_51d"

# Manifests
python scripts/make_run_manifest.py "${RUN_A}" --window-bars "$WIN" || true
python scripts/make_run_manifest.py "${RUN_B}" --window-bars "$WIN" || true

echo "== Smoke complete =="
