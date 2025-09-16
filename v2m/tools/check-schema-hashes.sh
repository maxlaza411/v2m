#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_FILE="schemas/.schema-hashes"
if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "::error::Missing schema hash baseline at $BASELINE_FILE" >&2
  exit 1
fi

actual="$(sha256sum schemas/*.json | sort)"
expected="$(<"$BASELINE_FILE")"

if [[ "$actual" == "$expected" ]]; then
  echo "Schema hashes unchanged."
  exit 0
fi

echo "::warning::Schema hashes changed." >&2

if [[ -n "${GITHUB_EVENT_PATH:-}" && -f "$GITHUB_EVENT_PATH" ]]; then
  labels=$(jq -r '.pull_request.labels[].name' "$GITHUB_EVENT_PATH" 2>/dev/null || echo "")
  if echo "$labels" | grep -Fxq "schema-change"; then
    echo "schema-change label present; allowing schema updates."
    exit 0
  fi
  echo "::error::Schema files changed. Add the 'schema-change' label to this PR." >&2
  exit 1
else
  echo "::error::Schema files changed outside of a labelled PR context." >&2
  exit 1
fi
