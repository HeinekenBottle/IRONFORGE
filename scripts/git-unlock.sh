#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

# Repo root check
git rev-parse --show-toplevel >/dev/null 2>&1 || { echo "Not inside a git repo"; exit 1; }

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "[git-unlock] repo: $ROOT"
if [ "$DRY_RUN" = "1" ]; then
  echo "[git-unlock] DRY RUN enabled — no changes will be made"
fi

# Abort in-progress ops (ignore if none)
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] git merge --abort | git rebase --abort | git cherry-pick --abort | git am --abort"
else
  git merge --abort 2>/dev/null || true
  git rebase --abort 2>/dev/null || true
  git cherry-pick --abort 2>/dev/null || true
  git am --abort 2>/dev/null || true
fi

# Kill processes holding .git
if command -v lsof >/dev/null 2>&1; then
  PIDS=$({ lsof -t +D .git 2>/dev/null || true; } | sort -u | tr '\n' ' ')
  if [ -n "${PIDS// }" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] would kill: $PIDS"
    else
      echo "[git-unlock] killing: $PIDS"
      kill -9 $PIDS || true
    fi
  fi
fi

# Remove lock files
echo "[git-unlock] scanning for *.lock under .git"
if [ "$DRY_RUN" = "1" ]; then
  # List lock files only
  find .git -maxdepth 3 -name "*.lock" -print || true
else
  echo "[git-unlock] removing *.lock under .git"
  find .git -maxdepth 3 -name "*.lock" -print -delete || true
fi

# Sanity status; if index is corrupt, rebuild it
if ! git status >/dev/null 2>&1; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would rebuild index: rm -f .git/index && git reset --quiet"
  else
    echo "[git-unlock] rebuilding index"
    rm -f .git/index
    git reset --quiet
  fi
fi

echo "[git-unlock] done."
