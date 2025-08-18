#!/usr/bin/env bash
set -euo pipefail
git config --global --unset maintenance.auto || true
git config --global --unset gc.auto || true
echo "[git-maintenance-on] Restored defaults (unset)."

