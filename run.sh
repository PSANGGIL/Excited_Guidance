#!/usr/bin/env bash
set -euo pipefail

# Single user-facing launcher for both training and prediction.
# Actual model logic remains in train.py and predict.py.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${MOLGUIDANCE_PYTHON:-python}"
LOG_DIR="${MOLGUIDANCE_LOG_DIR:-logs}"
MODE="nohup"

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh train [--foreground] [train.py options]
  ./run.sh pred  [--foreground] RUN_DIR_OR_CHECKPOINT [predict.py options]

Examples:
  ./run.sh train
  ./run.sh train --resume runs/<RUN_NAME>
  ./run.sh train --foreground --set training.batch_size=128

  ./run.sh pred runs/<RUN_NAME>
  ./run.sh pred runs/<RUN_NAME> --target 5.0 --n-mols 500
  ./run.sh pred --foreground runs/<RUN_NAME> --no-grid

Default behavior is nohup background execution.
Logs and PID files are written under logs/.
USAGE
}

has_option() {
    local wanted="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$wanted" || "$arg" == "$wanted="* ]]; then
            return 0
        fi
    done
    return 1
}

launch_command() {
    local job_name="$1"
    shift
    local -a cmd=("$@")

    mkdir -p "$LOG_DIR"
    local stamp
    stamp="$(date '+%Y%m%d_%H%M%S')_$$"
    local log_file="${LOG_DIR}/${job_name}_${stamp}.log"
    local pid_file="${LOG_DIR}/${job_name}_${stamp}.pid"

    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\nLog: %s\n' "$log_file"

    if [[ "$MODE" == "foreground" ]]; then
        "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        nohup "${cmd[@]}" > "$log_file" 2>&1 < /dev/null &
        local pid=$!
        printf '%s\n' "$pid" > "$pid_file"
        printf 'Started %s with PID %s\n' "$job_name" "$pid"
        printf 'PID file: %s\n' "$pid_file"
        printf 'Follow log: tail -f %q\n' "$log_file"
    fi
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

ACTION="$1"
shift

# --foreground is a launcher option and may appear anywhere after the action.
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--foreground" ]]; then
        MODE="foreground"
    else
        ARGS+=("$arg")
    fi
done

case "$ACTION" in
    train)
        CONFIG="${TRAIN_CONFIG:-config.example.yaml}"
        SEED="${TRAIN_SEED:-42}"

        CMD=("$PYTHON_BIN" -u train.py)

        # A resumed run reuses its saved config unless --config is explicitly given.
        if ! has_option --config "${ARGS[@]}" && ! has_option --resume "${ARGS[@]}"; then
            CMD+=(--config "$CONFIG")
        fi
        if ! has_option --seed "${ARGS[@]}"; then
            CMD+=(--seed "$SEED")
        fi
        CMD+=("${ARGS[@]}")

        launch_command "train" "${CMD[@]}"
        ;;

    pred|predict)
        TARGET_VALUE="${PRED_TARGET:-3.0}"
        N_MOLS="${PRED_N_MOLS:-100}"
        DEVICE="${PRED_DEVICE:-cuda:0}"
        SEED="${PRED_SEED:-42}"

        if [[ -n "${PRED_RUN_DIR:-}" ]]; then
            RUN_PATH="$PRED_RUN_DIR"
        else
            if [[ ${#ARGS[@]} -eq 0 || "${ARGS[0]}" == --* ]]; then
                echo "Error: prediction requires RUN_DIR_OR_CHECKPOINT." >&2
                usage >&2
                exit 2
            fi
            RUN_PATH="${ARGS[0]}"
            ARGS=("${ARGS[@]:1}")
        fi

        CMD=("$PYTHON_BIN" -u predict.py --run "$RUN_PATH")

        # Environment variables provide defaults; explicit CLI options win.
        if ! has_option --target "${ARGS[@]}" && ! has_option --property-values-file "${ARGS[@]}"; then
            CMD+=(--target "$TARGET_VALUE")
        fi
        if ! has_option --n-mols "${ARGS[@]}"; then
            CMD+=(--n-mols "$N_MOLS")
        fi
        if ! has_option --device "${ARGS[@]}"; then
            CMD+=(--device "$DEVICE")
        fi
        if ! has_option --seed "${ARGS[@]}"; then
            CMD+=(--seed "$SEED")
        fi
        CMD+=("${ARGS[@]}")

        RUN_LABEL="$(basename -- "$RUN_PATH")"
        RUN_LABEL="${RUN_LABEL%.ckpt}"
        launch_command "pred_${RUN_LABEL}" "${CMD[@]}"
        ;;

    -h|--help|help)
        usage
        ;;

    *)
        echo "Error: unknown action '$ACTION'. Use 'train' or 'pred'." >&2
        usage >&2
        exit 2
        ;;
esac
