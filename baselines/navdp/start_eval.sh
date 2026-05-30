#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
CHECKPOINT="${CHECKPOINT:-/home/monika/dyishere/project/MyResearch/NavDP/checkpoints/navdp/mini/checkpoint-52650navdp.ckpt}"

python navdp_server.py \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT"
