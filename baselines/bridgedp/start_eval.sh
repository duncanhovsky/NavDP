#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-/home/monika/dyishere/project/MyResearch/InternNav/checkpoints/bridgedp_train/ckpts/checkpoint-15927bridgedp.ckpt}"

python bridgedp_server.py \
    --port "$PORT" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --sigma_base 0.2 \
    --sigma_goal 0.01 \
    --sigma_floor 0.01 \
    --nogoal_front_distance 0.8 \
    --nogoal_sigma_start 0.03 \
    --nogoal_sigma_x_end 0.35 \
    --nogoal_sigma_y_end 0.80 \
    --nogoal_sigma_theta_end 0.60 \
    --nogoal_sigma_power 2.0 \
    --bridge_scale_invariant_sigma true \
    --bridge_anisotropic_xy true \
    --bridge_normal_sigma_ratio 1.0 \
    --bridge_tangent_sigma_ratio 0.05 \
    --bridge_theta_sigma_ratio 0.3 \
    --bridge_envelope_frontload 0.0 \
    --enable_trajectory_normalization true \
    --trajectory_norm_target_distance 2.0 \
    --trajectory_norm_min_distance_m 0.10 \
    --trajectory_norm_eps 1e-6 \
    --enable_scale_condition_token true \
    --scale_condition_clamp_min_m 0.10 \
    --scale_condition_clamp_max_m 20.0 \
    --enable_scale_rgbd_film true \
    --scale_rgbd_film_alpha 1.0 \
    --scale_rgbd_film_zero_init true \
    --scale_rgbd_film_use_layernorm true \
    --enable_goal_consistency_score true \
    --goal_consistency_terminal_weight 1.0 \
    --goal_consistency_path_weight 0.05 \
    --n_prior_tokens 4 \
    --num_train_timesteps 10 \
    --num_inference_timesteps 10 \
    --use_prior_traj false \
    --sample_num 16 \
    --exec_num_waypoints 24 \
    --exec_waypoint_spacing 0.15 \
    --enable_safety_layer true \
    --safety_clearance_m 0.25 \
    --safety_path_sample_spacing_m 0.05 \
    --retry_sigma_growth 1.5 \
    --max_retry_sigma_scale 3.0
