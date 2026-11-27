#!/usr/bin/env bash
# Fail when ablift/ changed since BASE but no changes/ fragment was added.
# Used by CI (lint job) and `make changelog-check`.
#
#   scripts/check_changelog_fragment.sh [BASE]   # BASE defaults to origin/main
#   SKIP_CHANGELOG=1 ...                         # skip (CI sets this from the no-changelog label)
set -euo pipefail

base="${1:-origin/main}"

if [ "${SKIP_CHANGELOG:-}" = "1" ]; then
  echo "SKIP_CHANGELOG=1: skipping the changelog fragment check."
  exit 0
fi

if ! git diff --name-only "$base"...HEAD | grep -q '^ablift/'; then
  echo "No changes under ablift/ since $base; no fragment needed."
  exit 0
fi

if uv run towncrier check --compare-with "$base"; then
  exit 0
fi

cat >&2 <<MSG

Changes under ablift/ need a changelog fragment (see changes/README.md):
  changes/<issue>.<type>.md  or  changes/+<slug>.<type>.md
  <type> is one of: added, changed, fixed, removed, docs

For internal changes with nothing to tell users, skip the check with
SKIP_CHANGELOG=1 locally, or the no-changelog label on the pull request.
MSG
exit 1
