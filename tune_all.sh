#!/usr/bin/env bash
# tune_all.sh — launches grid searches for all datasets in the background.
#
# Usage:
#   chmod +x tune_all.sh
#   ./tune_all.sh [--n_trials N] [--results_dir DIR]
#
# Datasets and grid size:
#   mushroom      12 combos  (hidden_width × n_blocks × mu_init)
#   heart         36 combos  (hidden_width × n_blocks × mu_init × prune)
#   monk          18 combos × 3 problems = 54 total
#   breast_cancer 36 combos  (hidden_width × n_blocks × mu_init × prune)
#
# Logs:
#   logs/tuning/<dataset>_<timestamp>.log
#
# To monitor:
#   tail -f logs/tuning/mushroom_*.log
#   tail -f logs/tuning/heart_*.log
#   tail -f logs/tuning/monk_*.log
#   tail -f logs/tuning/breast_cancer_*.log

set -euo pipefail

# ── Parameters (with defaults) ────────────────────────────────────────────────
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

# ── Logs ──────────────────────────────────────────────────────────────────────
LOG_DIR="logs/tuning"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_MUSHROOM="$LOG_DIR/mushroom_${TS}.log"
LOG_HEART="$LOG_DIR/heart_${TS}.log"
LOG_MONK="$LOG_DIR/monk_${TS}.log"
LOG_BREAST="$LOG_DIR/breast_cancer_${TS}.log"

# ── Python interpreter (prefers project .venv) ────────────────────────────────
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "================================================================"
echo "  ŁNN Residual — Grid Search (background)"
echo "  Trials/combo : $N_TRIALS"
echo "  Results dir  : $RESULTS_DIR"
echo "  Python       : $PYTHON"
echo "  Timestamp    : $TS"
echo "================================================================"

# ── Mushroom ──────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Mushroom (12 combos × $N_TRIALS trials = $((12 * N_TRIALS)) runs)"
nohup "$PYTHON" -u tuning/tune_mushroom.py \
    --n_trials    "$N_TRIALS" \
    --results_dir "$RESULTS_DIR" \
    > "$LOG_MUSHROOM" 2>&1 &
PID_MUSHROOM=$!
echo "      PID $PID_MUSHROOM  →  $LOG_MUSHROOM"

# ── Heart Disease ─────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Heart (36 combos × $N_TRIALS trials = $((36 * N_TRIALS)) runs)"
nohup "$PYTHON" -u tuning/tune_heart.py \
    --n_trials    "$N_TRIALS" \
    --results_dir "$RESULTS_DIR" \
    > "$LOG_HEART" 2>&1 &
PID_HEART=$!
echo "      PID $PID_HEART  →  $LOG_HEART"

# ── MONK (3 problems) ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] MONK (18 combos × 3 problems × $N_TRIALS trials = $((18 * 3 * N_TRIALS)) runs)"
nohup "$PYTHON" -u tuning/tune_monk.py \
    --problems 1 2 3 \
    --n_trials    "$N_TRIALS" \
    --results_dir "$RESULTS_DIR" \
    > "$LOG_MONK" 2>&1 &
PID_MONK=$!
echo "      PID $PID_MONK  →  $LOG_MONK"

# ── Breast Cancer ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Breast Cancer (36 combos × $N_TRIALS trials = $((36 * N_TRIALS)) runs)"
nohup "$PYTHON" -u tuning/tune_breast_cancer.py \
    --n_trials    "$N_TRIALS" \
    --results_dir "$RESULTS_DIR" \
    > "$LOG_BREAST" 2>&1 &
PID_BREAST=$!
echo "      PID $PID_BREAST  →  $LOG_BREAST"

# ── Monitoring instructions ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  To monitor:"
echo "    tail -f $LOG_MUSHROOM"
echo "    tail -f $LOG_HEART"
echo "    tail -f $LOG_MONK"
echo "    tail -f $LOG_BREAST"
echo ""
echo "  To check if still running:"
echo "    kill -0 $PID_MUSHROOM  2>/dev/null && echo 'mushroom:      running'"
echo "    kill -0 $PID_HEART     2>/dev/null && echo 'heart:         running'"
echo "    kill -0 $PID_MONK      2>/dev/null && echo 'monk:          running'"
echo "    kill -0 $PID_BREAST    2>/dev/null && echo 'breast_cancer: running'"
echo ""
echo "  To stop all:"
echo "    kill $PID_MUSHROOM $PID_HEART $PID_MONK $PID_BREAST"
echo "================================================================"

# Save PIDs for future reference
printf '%s\n' \
    "mushroom=$PID_MUSHROOM" \
    "heart=$PID_HEART" \
    "monk=$PID_MONK" \
    "breast_cancer=$PID_BREAST" \
    > "$LOG_DIR/pids_${TS}.txt"
echo "  PIDs saved to $LOG_DIR/pids_${TS}.txt"
echo ""
