#!/usr/bin/env bash
if [ ! -d "ironforge" ]; then
  echo "ERROR: ironforge/ package is missing. CI and tests depend on it."
  exit 1
fi