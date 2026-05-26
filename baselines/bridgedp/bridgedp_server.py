"""Bridge-DP Flask 服务端。

独立实现，暴露与 NavDP 完全一致的 HTTP API 接口，
使得评估脚本无需任何修改即可调用。

API 路由：
- POST /navigator_reset      : 初始化 Agent
- POST /navigator_reset_env  : 重置单个环境
- POST /pointgoal_step       : PointGoal 推理
- POST /nogoal_step           : NoGoal 推理
- POST /imagegoal_step        : ImageGoal 推理
- POST /pixelgoal_step        : PixelGoal 推理
"""

import argparse
import datetime
import json
import os
import time

import cv2
import imageio
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from policy_agent import BridgeDP_Agent


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888)
parser.add_argument("--checkpoint", type=str, default="./bridgedp.ckpt")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--sigma_base", type=float, default=1.0)
parser.add_argument("--sigma_goal", type=float, default=0.1)
parser.add_argument("--sigma_floor", type=float, default=None)
parser.add_argument("--nogoal_front_distance", type=float, default=0.8)
parser.add_argument("--nogoal_sigma_start", type=float, default=0.03)
parser.add_argument("--nogoal_sigma_x_end", type=float, default=0.35)
parser.add_argument("--nogoal_sigma_y_end", type=float, default=0.80)
parser.add_argument("--nogoal_sigma_theta_end", type=float, default=0.60)
parser.add_argument("--nogoal_sigma_power", type=float, default=2.0)
parser.add_argument("--bridge_scale_invariant_sigma", type=str2bool, default=False)
parser.add_argument("--bridge_anisotropic_xy", type=str2bool, default=True)
parser.add_argument("--bridge_normal_sigma_ratio", type=float, default=0.25)
parser.add_argument("--bridge_tangent_sigma_ratio", type=float, default=0.03)
parser.add_argument("--bridge_theta_sigma_ratio", type=float, default=0.05)
parser.add_argument("--bridge_envelope_frontload", type=float, default=0.0)
parser.add_argument("--n_prior_tokens", type=int, default=4)
parser.add_argument("--enable_trajectory_normalization", type=str2bool, default=False)
parser.add_argument("--trajectory_norm_target_distance", type=float, default=2.0)
parser.add_argument("--trajectory_norm_min_distance_m", type=float, default=0.10)
parser.add_argument("--trajectory_norm_eps", type=float, default=1e-6)
parser.add_argument("--enable_scale_condition_token", type=str2bool, default=False)
parser.add_argument("--scale_condition_clamp_min_m", type=float, default=0.10)
parser.add_argument("--scale_condition_clamp_max_m", type=float, default=20.0)
parser.add_argument("--enable_scale_rgbd_film", type=str2bool, default=False)
parser.add_argument("--scale_rgbd_film_alpha", type=float, default=1.0)
parser.add_argument("--scale_rgbd_film_zero_init", type=str2bool, default=True)
parser.add_argument("--scale_rgbd_film_use_layernorm", type=str2bool, default=True)
parser.add_argument("--enable_goal_consistency_score", type=str2bool, default=False)
parser.add_argument("--goal_consistency_terminal_weight", type=float, default=1.0)
parser.add_argument("--goal_consistency_path_weight", type=float, default=0.2)
parser.add_argument("--num_train_timesteps", type=int, default=100)
parser.add_argument("--num_inference_timesteps", type=int, default=100)
parser.add_argument("--use_prior_traj", type=str2bool, default=False)
parser.add_argument("--sample_num", type=int, default=16)
parser.add_argument("--exec_num_waypoints", type=int, default=24)
parser.add_argument("--exec_waypoint_spacing", type=float, default=0.15)
args = parser.parse_known_args()[0]

app = Flask(__name__)
bridgedp_navigator = None
bridgedp_fps_writer = None


@app.route("/navigator_reset", methods=['POST'])
def bridgedp_reset():
    global bridgedp_navigator, bridgedp_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))

    if bridgedp_navigator is None:
        bridgedp_navigator = BridgeDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            sigma_base=args.sigma_base,
            sigma_goal=args.sigma_goal,
            sigma_floor=args.sigma_floor,
            nogoal_front_distance=args.nogoal_front_distance,
            nogoal_sigma_start=args.nogoal_sigma_start,
            nogoal_sigma_x_end=args.nogoal_sigma_x_end,
            nogoal_sigma_y_end=args.nogoal_sigma_y_end,
            nogoal_sigma_theta_end=args.nogoal_sigma_theta_end,
            nogoal_sigma_power=args.nogoal_sigma_power,
            bridge_scale_invariant_sigma=args.bridge_scale_invariant_sigma,
            bridge_anisotropic_xy=args.bridge_anisotropic_xy,
            bridge_normal_sigma_ratio=args.bridge_normal_sigma_ratio,
            bridge_tangent_sigma_ratio=args.bridge_tangent_sigma_ratio,
            bridge_theta_sigma_ratio=args.bridge_theta_sigma_ratio,
            bridge_envelope_frontload=args.bridge_envelope_frontload,
            n_prior_tokens=args.n_prior_tokens,
            enable_trajectory_normalization=args.enable_trajectory_normalization,
            trajectory_norm_target_distance=args.trajectory_norm_target_distance,
            trajectory_norm_min_distance_m=args.trajectory_norm_min_distance_m,
            trajectory_norm_eps=args.trajectory_norm_eps,
            enable_scale_condition_token=args.enable_scale_condition_token,
            scale_condition_clamp_min_m=args.scale_condition_clamp_min_m,
            scale_condition_clamp_max_m=args.scale_condition_clamp_max_m,
            enable_scale_rgbd_film=args.enable_scale_rgbd_film,
            scale_rgbd_film_alpha=args.scale_rgbd_film_alpha,
            scale_rgbd_film_zero_init=args.scale_rgbd_film_zero_init,
            scale_rgbd_film_use_layernorm=args.scale_rgbd_film_use_layernorm,
            enable_goal_consistency_score=args.enable_goal_consistency_score,
            goal_consistency_terminal_weight=args.goal_consistency_terminal_weight,
            goal_consistency_path_weight=args.goal_consistency_path_weight,
            num_train_timesteps=args.num_train_timesteps,
            num_inference_timesteps=args.num_inference_timesteps,
            use_prior_traj=args.use_prior_traj,
            sample_num=args.sample_num,
            exec_num_waypoints=args.exec_num_waypoints,
            exec_waypoint_spacing=args.exec_waypoint_spacing,
            navi_model=args.checkpoint,
            device=args.device,
        )
        bridgedp_navigator.reset(batchsize, threshold)
    else:
        bridgedp_navigator.reset(batchsize, threshold)

    if bridgedp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d_%H-%M-%S")
        bridgedp_fps_writer = imageio.get_writer(
            "{}_fps_pointgoal.mp4".format(format_time), fps=7
        )
    else:
        bridgedp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d_%H-%M-%S")
        bridgedp_fps_writer = imageio.get_writer(
            "{}_fps_pointgoal.mp4".format(format_time), fps=7
        )

    return jsonify({"algo": "bridgedp"})


@app.route("/navigator_reset_env", methods=['POST'])
def bridgedp_reset_env():
    global bridgedp_navigator
    bridgedp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo": "bridgedp"})


@app.route("/pointgoal_step", methods=['POST'])
def bridgedp_step_pointgoal():
    global bridgedp_navigator, bridgedp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = bridgedp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        bridgedp_navigator.step_pointgoal(goal, image, depth)
    phase3_time = time.time()
    bridgedp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    early_steps = max(1, all_trajectory.shape[-2] // 4)
    early_lateral_spread = np.std(all_trajectory[:, :, :early_steps, 1])
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    # ── 诊断打印 ──
    print(
        f"[BridgeDP diag] goal={goal[0]}, "
        f"traj_x=[{execute_trajectory[:,:,0].min():.4f}, {execute_trajectory[:,:,0].max():.4f}], "
        f"traj_y=[{execute_trajectory[:,:,1].min():.4f}, {execute_trajectory[:,:,1].max():.4f}], "
        f"early_y_std={early_lateral_spread:.4f}, "
        f"critic=[{all_values.min():.4f}, {all_values.max():.4f}], "
        f"traj_shape={execute_trajectory.shape}"
    )

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
    })


@app.route("/nogoal_step", methods=['POST'])
def bridgedp_step_nogoal():
    global bridgedp_navigator, bridgedp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = bridgedp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        bridgedp_navigator.step_nogoal(image, depth)
    phase3_time = time.time()
    bridgedp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    # ── 诊断打印 ──
    print(
        f"[BridgeDP diag nogoal] "
        f"traj_x=[{execute_trajectory[:,:,0].min():.4f}, {execute_trajectory[:,:,0].max():.4f}], "
        f"traj_y=[{execute_trajectory[:,:,1].min():.4f}, {execute_trajectory[:,:,1].max():.4f}], "
        f"critic=[{all_values.min():.4f}, {all_values.max():.4f}], "
        f"traj_shape={execute_trajectory.shape}"
    )

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
    })


@app.route("/imagegoal_step", methods=['POST'])
def bridgedp_step_imagegoal():
    global bridgedp_navigator, bridgedp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    batch_size = bridgedp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        bridgedp_navigator.step_imagegoal(goal, image, depth)
    phase3_time = time.time()
    bridgedp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
    })


@app.route("/pixelgoal_step", methods=['POST'])
def bridgedp_step_pixelgoal():
    global bridgedp_navigator, bridgedp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y), axis=1)
    batch_size = bridgedp_navigator.batch_size

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        bridgedp_navigator.step_pixelgoal(goal, image, depth)
    phase3_time = time.time()
    bridgedp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist(),
    })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=args.port)
