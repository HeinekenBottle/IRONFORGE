#!/usr/bin/env bash
set -euo pipefail
git config --global maintenance.auto false
git config --global gc.auto 0
echo "[git-maintenance-off] Disabled background maintenance and auto-gc globally."

