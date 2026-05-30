#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
PORT="${PORT:-8888}"
N="${N:-${1:-100}}"
NUM_ENVS="${NUM_ENVS:-1}"
ASSET_ROOT="${ASSET_ROOT:-$SCRIPT_DIR/assets/scenes}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/eval_pointgoal_wheeled.py}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "N must be a positive integer, got: $N" >&2
    exit 1
fi

has_pointgoal_episode() {
    local candidate="$1"
    local usd_path=""
    local init_path=""

    usd_path="$(find "$candidate" -maxdepth 1 -name "*.usd" ! -name "*noMDL*" -print -quit)"
    init_path="$(find "$candidate" -maxdepth 1 -type f -name "*pointgoal*.npy" -print -quit)"

    [ -n "$usd_path" ] && [ -n "$init_path" ]
}

resolve_scene_dir() {
    local group="$1"
    local nested="${2:-}"
    local candidates=()

    if [ -n "$nested" ]; then
        candidates+=("$ASSET_ROOT/$group/$nested")
    fi
    candidates+=("$ASSET_ROOT/$group")

    if [ "$group" = "internscene_home" ]; then
        if [ -n "$nested" ]; then
            candidates+=("$ASSET_ROOT/internscenes_home/$nested")
        fi
        candidates+=("$ASSET_ROOT/internscenes_home")
    fi

    for candidate in "${candidates[@]}"; do
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

collect_sequences() {
    local scene_dir="$1"
    local candidate

    while IFS= read -r -d '' candidate; do
        if has_pointgoal_episode "$candidate"; then
            printf '%s\0' "$candidate"
        fi
    done < <(find "$scene_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
}

run_scene_group() {
    local group="$1"
    local scale="$2"
    local nested="${3:-}"
    local scene_dir
    local sequences=()
    local idx
    local total

    if ! scene_dir="$(resolve_scene_dir "$group" "$nested")"; then
        echo "Skip $group: directory not found under $ASSET_ROOT" >&2
        return 0
    fi

    while IFS= read -r -d '' sequence; do
        sequences+=("$sequence")
    done < <(collect_sequences "$scene_dir")

    total="${#sequences[@]}"
    if [ "$total" -eq 0 ]; then
        echo "Skip $group: no pointgoal sequences found in $scene_dir" >&2
        return 0
    fi

    echo "Running $group: $total sequences, $N episodes each, scale=$scale, port=$PORT"
    for idx in "${!sequences[@]}"; do
        echo "[$group] $((idx + 1))/$total $(basename "${sequences[$idx]}")"
        "$PYTHON" "$EVAL_SCRIPT" \
            --port "$PORT" \
            --scene_dir "$scene_dir" \
            --scene_index "$idx" \
            --scene_scale "$scale" \
            --num_episodes "$N" \
            --num_envs "$NUM_ENVS"
    done
}

run_scene_group "cluttered_easy" "1.0"
run_scene_group "cluttered_hard" "1.0"
run_scene_group "internscenes_commercial" "0.01" "scenes_commercial"
run_scene_group "internscene_home" "0.01" "scenes_home"
