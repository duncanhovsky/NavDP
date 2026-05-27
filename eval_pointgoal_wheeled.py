import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./asset_scenes/cluttered_easy")
parser.add_argument(
    "--scene_name", type=str, default=None,
    help="Exact scene subdirectory name; overrides --scene_index.")
parser.add_argument(
    "--scene_index", type=int, default=8)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--speed", type=float, default=0.5)
parser.add_argument(
    "--port", type=int, default=8888)
args_cli = parser.parse_args()
if args_cli.num_episodes < 1:
    parser.error("--num_episodes must be a positive integer")
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset,pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller

planning_input = PlanningInput() 
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
episode_generation = np.zeros((args_cli.num_envs,), dtype=np.int64)
SAFETY_CLEARANCE_M = 0.25
SAFETY_PATH_SAMPLE_SPACING_M = 0.05
SAFETY_PROJECTION_HEIGHT_M = -0.20
SAFETY_HEIGHT_BAND_PX = 8


def depth_to_local_obstacles(depth, intrinsic):
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    height, width = depth.shape
    stride = 2
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    forward = depth[::stride, ::stride]
    expected_v = (
        height - 1
        - fy * SAFETY_PROJECTION_HEIGHT_M / np.maximum(forward, 1e-6)
        - cy
    )
    valid = (
        np.isfinite(forward)
        & (forward > 0.1)
        & (forward < 5.0)
        & (np.abs(vv - expected_v) <= SAFETY_HEIGHT_BAND_PX)
    )
    if not valid.any():
        return np.zeros((0, 2), dtype=np.float32)
    lateral = -(uu[valid] - cx) * forward[valid] / max(fx, 1e-6)
    return np.stack([forward[valid], lateral], axis=-1).astype(np.float32)


def local_path_min_clearance(path_local, obstacle_points):
    if obstacle_points.shape[0] == 0:
        return np.inf
    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), np.asarray(path_local, dtype=np.float32)[:, :2]],
        axis=0,
    )
    dense = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(length / SAFETY_PATH_SAMPLE_SPACING_M)))
        alpha = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:]
        dense.extend(start[None, :] + alpha[:, None] * (end - start)[None, :])
    dense = np.asarray(dense, dtype=np.float32)
    return np.linalg.norm(
        dense[:, None, :] - obstacle_points[None, :, :], axis=-1
    ).min()


def remaining_world_path_is_safe(path_world, camera_pos, camera_rot, depth, intrinsic):
    if path_world is None or len(path_world) == 0:
        return False
    nearest = int(np.argmin(np.linalg.norm(path_world - camera_pos[:2], axis=-1)))
    remaining = path_world[nearest:]
    if remaining.shape[0] == 0:
        return False
    delta_world = np.zeros((remaining.shape[0], 3), dtype=np.float32)
    delta_world[:, :2] = remaining - camera_pos[:2]
    local = (camera_rot.T @ delta_world.T).T[:, :2]
    local = local[local[:, 0] >= -0.05]
    if local.shape[0] == 0:
        return False
    obstacles = depth_to_local_obstacles(depth, intrinsic)
    return local_path_min_clearance(local, obstacles) >= SAFETY_CLEARANCE_M

def planning_thread(env, camera_intrinsic):
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
                request_generation = episode_generation.copy()
            with output_lock:
                planning_output.is_planning = True
            
            # Start timing planning
            planning_start = time.time()
            (trajectory_points_camera, all_trajectories_camera,
             all_values_camera, response_metadata) = pointgoal_step(
                goal, image, depth, port=args_cli.port, return_metadata=True
            )
            safety_metadata = response_metadata.get("safety", {})
            safety_status = safety_metadata.get(
                "status", ["accepted_without_safety_status"] * trajectory_points_camera.shape[0]
            )
            intrinsic_np = camera_intrinsic.cpu().numpy()
            with input_lock:
                stale_request = request_generation != episode_generation
            with output_lock:
                prior_paths = (
                    None if planning_output.trajectory_points_world is None
                    else planning_output.trajectory_points_world.copy()
                )
                prior_valid = (
                    np.zeros((trajectory_points_camera.shape[0],), dtype=bool)
                    if planning_output.valid_safe_plan is None
                    else planning_output.valid_safe_plan.copy()
                )
            # Transform trajectory from camera frame to world frame
            batch_optimal_points_world = []
            batch_mpc_controllers = []
            safety_stop_flags = []
            valid_safe_plan = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                blocked = safety_status[idx] == "blocked_no_safe_candidate"
                if stale_request[idx]:
                    blocked = True
                    safety_status[idx] = "discarded_after_episode_reset"
                hold_safe = (
                    blocked
                    and not stale_request[idx]
                    and prior_paths is not None
                    and bool(prior_valid[idx])
                    and remaining_world_path_is_safe(
                        prior_paths[idx], camera_pos[idx], camera_rot[idx], depth[idx], intrinsic_np
                    )
                )
                if hold_safe:
                    trajectory_points_world = prior_paths[idx].copy()
                    safety_status[idx] = "holding_last_safe_trajectory"
                elif blocked:
                    trajectory_points_world = np.repeat(
                        camera_pos[idx, :2][None, :],
                        trajectory_points_camera.shape[1],
                        axis=0,
                    )
                    if not stale_request[idx]:
                        safety_status[idx] = "stopped_no_safe_trajectory"
                batch_optimal_points_world.append(trajectory_points_world)
                safety_stop_flags.append(bool(blocked and not hold_safe))
                valid_safe_plan.append(bool(not blocked or hold_safe))
                batch_mpc_controllers.append(
                    MPC_Controller(
                        trajectory_points_world,
                        desired_v=args_cli.speed,
                        v_max=args_cli.speed,
                        w_max=args_cli.speed,
                    )
                )
            batch_optimal_points_world = np.array(batch_optimal_points_world)
           
            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                # Transform all trajectories
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            # Update shared state
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.mpc_controllers = batch_mpc_controllers
                planning_output.safety_stop_flags = np.asarray(safety_stop_flags, dtype=bool)
                planning_output.safety_status = safety_status
                planning_output.valid_safe_plan = np.asarray(valid_safe_plan, dtype=bool)
                planning_output.is_planning = False
                planning_output.planning_error = None
            
            # Print planning timing
            planning_time = time.time() - planning_start
            # print(f"Planning time: {planning_time:.3f}s, Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

scene_name = args_cli.scene_name
if scene_name is None:
    scene_name = os.listdir(args_cli.scene_dir)[args_cli.scene_index]
scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
if not os.path.isdir(scene_path):
    raise FileNotFoundError(f"Scene sequence directory does not exist: {scene_path}")
usd_path,init_path = find_usd_path(scene_path,task='pointgoal')
scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal = GOAL_CFG
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path":init_path, 
                                       'height_offset':0.1,
                                       'robot_visible': False,
                                       'light_enabled': False}
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
_,infos = env.reset()
# warm-up
PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
    
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control", 
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]

trajectory_length = np.zeros((scene_config.num_envs))
episode_collisions = np.zeros((scene_config.num_envs), dtype=bool)

while simulation_app.is_running():
    with torch.inference_mode():
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        
        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        # based on the current world trajectory 
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :2].norm(dim=-1).cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[:, 2].cpu().numpy()

        x0 = np.stack(
            [camera_pos[:, 0], camera_pos[:, 1], np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
             robot_vel, robot_ang_vel],
            axis=-1,
        )
        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        current_mpcs = None
        safety_stop_flags = None
        safety_status = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
                current_mpcs = planning_output.mpc_controllers
                safety_stop_flags = planning_output.safety_stop_flags.copy()
                safety_status = list(planning_output.safety_status)
        
        if current_trajectory is not None:
            control_start = time.time()
            action_list = []
            for i in range(args_cli.num_envs):
                vis_image = vis_manager[i].visualize_trajectory(
                    images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )
                force_stop = safety_stop_flags is not None and bool(safety_stop_flags[i])
                local_mpc = None if current_mpcs is None else current_mpcs[i]
                if force_stop or local_mpc is None:
                    v, w = 0.0, 0.0
                else:
                    t0 = time.time()
                    opt_u_controls, opt_x_states = local_mpc.solve(x0[i, :3])
                    print(f"solve mpc cost {time.time() - t0}")
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)
                
                try:
                    vis_image = draw_box_with_text(vis_image,0,0,430,50,"desired lin.:%.2f ang.:%.2f"%(v,w))
                    vis_image = draw_box_with_text(vis_image,0,50,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel[i],robot_ang_vel[i]))
                    if current_all_values is not None:
                        vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values[i]), np.min(current_all_values[i])))
                    vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                    if safety_status is not None:
                        vis_image = draw_box_with_text(vis_image,0,100,430,50,"safety:%s"%safety_status[i])
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
                
            action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            # Get actual joint velocities from Isaac Sim
            actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
            desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
            trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
        else:
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            print("No trajectory available, using zero action")

        try:
            forces = env.unwrapped.scene.sensors['contact_sensor'].data.net_forces_w.cpu().numpy()
            episode_collisions |= np.linalg.norm(forces, axis=-1).max(axis=-1) > DINGO_THRESHOLD
        except Exception:
            pass
        
        for i in range(args_cli.num_envs):
            if dones[i] == True and len(evaluation_metrics) < args_cli.num_episodes:
                episode_num += 1
                with input_lock:
                    episode_generation[i] += 1
                navigator_reset(env_id=i,port=args_cli.port)
                with output_lock:
                    if planning_output.valid_safe_plan is not None:
                        planning_output.valid_safe_plan[i] = False
                    if planning_output.safety_stop_flags is not None:
                        planning_output.safety_stop_flags[i] = True
                success_flag = (np.sqrt(np.square(goals[i]).sum())<1.5).astype(np.float32)
                fps_writer[i].close()
                evaluation_metrics.append({'success':success_flag,
                                           'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                           'distance':euclidean[i],
                                           'collision': float(episode_collisions[i])})
                write_metrics(evaluation_metrics,save_dir+"metric.csv")
                euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0
                episode_collisions[i] = False
        
        if len(evaluation_metrics) >= args_cli.num_episodes:
            break
       
                
   

        
