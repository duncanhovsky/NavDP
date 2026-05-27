#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
SCENE_SCALE="${SCENE_SCALE:-1.0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENES_ROOT="${SCENES_ROOT:-$SCRIPT_DIR/assets/scenes}"

if ! [[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EPISODES must be a positive integer, got: $NUM_EPISODES" >&2
    exit 2
fi

run_scene_group() {
    local group_name="$1"
    local sequence_prefix="$2"
    local scene_dir="$SCENES_ROOT/$group_name"

    if [[ ! -d "$scene_dir" ]]; then
        echo "Scene group directory does not exist: $scene_dir" >&2
        exit 1
    fi

    for scene_number in {0..9}; do
        local scene_name="${sequence_prefix}_${scene_number}"
        if [[ ! -d "$scene_dir/$scene_name" ]]; then
            echo "Scene sequence directory does not exist: $scene_dir/$scene_name" >&2
            exit 1
        fi

        echo "[eval] ${group_name}/${scene_name}: ${NUM_EPISODES} episodes"
        "$PYTHON_BIN" "$SCRIPT_DIR/eval_pointgoal_wheeled.py" \
            --port "$PORT" \
            --scene_dir "$scene_dir" \
            --scene_name "$scene_name" \
            --scene_scale "$SCENE_SCALE" \
            --num_episodes "$NUM_EPISODES"
    done
}

run_scene_group "cluttered_easy" "easy"
run_scene_group "cluttered_hard" "hard"
