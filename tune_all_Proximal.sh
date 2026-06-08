#!/usr/bin/env bash
# tune_all_Proximal.sh — launches the Proximal grid search for all datasets in the background.
#
# Usage:
#   chmod +x tune_all_Proximal.sh
#   ./tune_all_Proximal.sh [--n_trials N] [--results_dir DIR]
#
# Grid per dataset (18 combos × N trials):
#   hidden_layers  ∈ {[4,4], [6,4], [8,4]}
#   lr             ∈ {5e-3, 1e-2, 2e-2}
#   lambda_sparse  ∈ {1e-4, 1e-3}
#
# Note: Proximal needs more iterations than LM/STE — max_iter=5000-8000.
#       Each combo may take 20-60s; estimate 2-4h per dataset.
#
# Logs:
#   logs/tuning/proximal_<dataset>_<timestamp>.log

set -euo pipefail

# ── Parameters ────────────────────────────────────────────────────────────────
N_TRIALS=5
RESULTS_DIR="results/tuning"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_trials)    N_TRIALS="$2";    shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Root directory ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python interpreter (prefers project .venv) ────────────────────────────────
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

# ── Logs ──────────────────────────────────────────────────────────────────────
LOG_DIR="logs/tuning"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

COMBOS=18

echo "================================================================"
echo "  ŁNN — Proximal Grid Search (background)"
echo "  Trials/combo : $N_TRIALS"
echo "  Combos/ds    : $COMBOS  (hidden_layers × lr × lambda_sparse)"
echo "  Results dir  : $RESULTS_DIR"
echo "  Python       : $PYTHON"
echo "  Timestamp    : $TS"
echo "================================================================"

declare -A PIDS

# ── Launch one process per dataset ────────────────────────────────────────────
i=0
for DS in mushroom heart monk_1 monk_2 monk_3 breast_cancer; do
    i=$((i+1))
    LOG="$LOG_DIR/proximal_${DS}_${TS}.log"
    RUNS=$((COMBOS * N_TRIALS))
    echo ""
    echo "[$i/6] Proximal — $DS  ($COMBOS combos × $N_TRIALS trials = $RUNS runs)"
    nohup "$PYTHON" -u tuning/tune_proximal.py \
        --dataset     "$DS" \
        --n_trials    "$N_TRIALS" \
        --results_dir "$RESULTS_DIR" \
        > "$LOG" 2>&1 &
    PIDS[$DS]=$!
    echo "      PID ${PIDS[$DS]}  →  $LOG"
done

# ── Instructions ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  To monitor (examples):"
echo "    tail -f $LOG_DIR/proximal_mushroom_${TS}.log"
echo "    tail -f $LOG_DIR/proximal_heart_${TS}.log"
echo "    tail -f $LOG_DIR/proximal_monk_1_${TS}.log"
echo ""
echo "  To check active processes:"
for DS in mushroom heart monk_1 monk_2 monk_3 breast_cancer; do
    echo "    kill -0 ${PIDS[$DS]} 2>/dev/null && echo 'proximal_$DS: running'"
done
echo ""
echo "  To stop all:"
echo "    kill ${PIDS[mushroom]} ${PIDS[heart]} ${PIDS[monk_1]} ${PIDS[monk_2]} ${PIDS[monk_3]} ${PIDS[breast_cancer]}"
echo "================================================================"

# Save PIDs
{
    for DS in mushroom heart monk_1 monk_2 monk_3 breast_cancer; do
        echo "proximal_$DS=${PIDS[$DS]}"
    done
} > "$LOG_DIR/pids_proximal_${TS}.txt"
echo "  PIDs saved to $LOG_DIR/pids_proximal_${TS}.txt"
echo ""
