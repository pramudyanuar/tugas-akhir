#!/usr/bin/env bash
set -euo pipefail

# Project cleanup utility.
# Default mode is dry-run; use --apply to actually delete files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=1
CLEAN_OUTPUTS=1
CLEAN_CACHES=1

usage() {
  cat <<'EOF'
Usage: scripts/clean.sh [options]

Options:
  --apply           Actually delete files/directories (default is dry-run)
  --dry-run         Show what would be deleted (default)
  --no-outputs      Do not clean generated outputs (logs/visualizations)
  --no-caches       Do not clean Python/test caches
  -h, --help        Show this help message

Examples:
  scripts/clean.sh
  scripts/clean.sh --apply
  scripts/clean.sh --apply --no-outputs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      DRY_RUN=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-outputs)
      CLEAN_OUTPUTS=0
      shift
      ;;
    --no-caches)
      CLEAN_CACHES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Build cleanup target list.
declare -a TARGETS=()

if [[ "$CLEAN_CACHES" -eq 1 ]]; then
  while IFS= read -r path; do TARGETS+=("$path"); done < <(
    find "$ROOT_DIR" \( -path "$ROOT_DIR/venv" -o -path "$ROOT_DIR/.venv" -o -path "$ROOT_DIR/.git" \) -prune -o -type d \( \
      -name '__pycache__' -o \
      -name '.pytest_cache' -o \
      -name '.mypy_cache' -o \
      -name '.ruff_cache' -o \
      -name '.ipynb_checkpoints' \
    \) -print 2>/dev/null
  )
fi

if [[ "$CLEAN_OUTPUTS" -eq 1 ]]; then
  for rel in \
    "logs/evaluation" \
    "logs/training" \
    "logs/example_training" \
    "visualizations" \
    "visualizations_optimized"; do
    abs="$ROOT_DIR/$rel"
    if [[ -e "$abs" ]]; then
      TARGETS+=("$abs")
    fi
  done
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No cleanup targets found."
  exit 0
fi

echo "Cleanup root: $ROOT_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Mode: DRY-RUN (no files will be deleted)"
else
  echo "Mode: APPLY (files/directories will be deleted)"
fi

echo
echo "Targets:"
for t in "${TARGETS[@]}"; do
  echo " - $t"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

for t in "${TARGETS[@]}"; do
  rm -rf "$t"
done

echo
echo "Cleanup completed."
