"""Bridge-DP agent wrapper for the NavDP evaluation server.

Bridge-DP predicts raw absolute curve control points.  NavDP's evaluator feeds
returned points into an MPC reference tracker, so this wrapper keeps two
representations separate:

* raw curve points: used as Bridge-DP prior on the next planning step;
* execution waypoints: arc-length-resampled points returned to the evaluator.
"""

import os

import cv2
import numpy as np
import torch

from policy_network import BridgeDP_Policy


class BridgeDP_Agent:
    def __init__(
        self,
        image_intrinsic,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        sigma_base=1.0,
        sigma_goal=0.1,
        sigma_floor=None,
        nogoal_front_distance=0.8,
        nogoal_sigma_start=0.03,
        nogoal_sigma_x_end=0.35,
        nogoal_sigma_y_end=0.80,
        nogoal_sigma_theta_end=0.60,
        nogoal_sigma_power=2.0,
        bridge_scale_invariant_sigma=False,
        bridge_anisotropic_xy=True,
        bridge_normal_sigma_ratio=0.25,
        bridge_tangent_sigma_ratio=0.03,
        bridge_theta_sigma_ratio=0.05,
        bridge_envelope_frontload=0.0,
        n_prior_tokens=4,
        enable_trajectory_normalization=False,
        trajectory_norm_target_distance=2.0,
        trajectory_norm_min_distance_m=0.10,
        trajectory_norm_eps=1e-6,
        enable_scale_condition_token=False,
        scale_condition_clamp_min_m=0.10,
        scale_condition_clamp_max_m=20.0,
        enable_scale_rgbd_film=False,
        scale_rgbd_film_alpha=1.0,
        scale_rgbd_film_zero_init=True,
        scale_rgbd_film_use_layernorm=True,
        enable_goal_consistency_score=False,
        goal_consistency_terminal_weight=1.0,
        goal_consistency_path_weight=0.2,
        num_train_timesteps=100,
        num_inference_timesteps=100,
        use_prior_traj=False,
        sample_num=16,
        exec_num_waypoints=24,
        exec_waypoint_spacing=0.15,
        enable_safety_layer=True,
        safety_clearance_m=0.25,
        safety_path_sample_spacing_m=0.05,
        safety_depth_max_m=5.0,
        safety_projection_height_m=-0.20,
        safety_height_band_px=8,
        retry_sigma_growth=1.5,
        max_retry_sigma_scale=3.0,
        navi_model="./bridgedp.ckpt",
        device='cuda:0',
    ):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.sample_num = sample_num
        self.exec_num_waypoints = exec_num_waypoints
        self.exec_waypoint_spacing = float(exec_waypoint_spacing)
        self.enable_safety_layer = bool(enable_safety_layer)
        self.safety_clearance_m = float(safety_clearance_m)
        self.safety_path_sample_spacing_m = float(safety_path_sample_spacing_m)
        self.safety_depth_max_m = float(safety_depth_max_m)
        self.safety_projection_height_m = float(safety_projection_height_m)
        self.safety_height_band_px = int(safety_height_band_px)
        self.retry_sigma_growth = float(retry_sigma_growth)
        self.max_retry_sigma_scale = float(max_retry_sigma_scale)

        self.navi_former = BridgeDP_Policy(
            image_size=image_size,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
            sigma_base=sigma_base,
            sigma_goal=sigma_goal,
            sigma_floor=sigma_floor,
            nogoal_front_distance=nogoal_front_distance,
            nogoal_sigma_start=nogoal_sigma_start,
            nogoal_sigma_x_end=nogoal_sigma_x_end,
            nogoal_sigma_y_end=nogoal_sigma_y_end,
            nogoal_sigma_theta_end=nogoal_sigma_theta_end,
            nogoal_sigma_power=nogoal_sigma_power,
            bridge_scale_invariant_sigma=bridge_scale_invariant_sigma,
            bridge_anisotropic_xy=bridge_anisotropic_xy,
            bridge_normal_sigma_ratio=bridge_normal_sigma_ratio,
            bridge_tangent_sigma_ratio=bridge_tangent_sigma_ratio,
            bridge_theta_sigma_ratio=bridge_theta_sigma_ratio,
            bridge_envelope_frontload=bridge_envelope_frontload,
            n_prior_tokens=n_prior_tokens,
            enable_trajectory_normalization=enable_trajectory_normalization,
            trajectory_norm_target_distance=trajectory_norm_target_distance,
            trajectory_norm_min_distance_m=trajectory_norm_min_distance_m,
            trajectory_norm_eps=trajectory_norm_eps,
            enable_scale_condition_token=enable_scale_condition_token,
            scale_condition_clamp_min_m=scale_condition_clamp_min_m,
            scale_condition_clamp_max_m=scale_condition_clamp_max_m,
            enable_scale_rgbd_film=enable_scale_rgbd_film,
            scale_rgbd_film_alpha=scale_rgbd_film_alpha,
            scale_rgbd_film_zero_init=scale_rgbd_film_zero_init,
            scale_rgbd_film_use_layernorm=scale_rgbd_film_use_layernorm,
            enable_goal_consistency_score=enable_goal_consistency_score,
            goal_consistency_terminal_weight=goal_consistency_terminal_weight,
            goal_consistency_path_weight=goal_consistency_path_weight,
            num_train_timesteps=num_train_timesteps,
            num_inference_timesteps=num_inference_timesteps,
            use_prior_traj=use_prior_traj,
            device=device,
        )
        self._load_checkpoint(navi_model)
        self.navi_former.to(self.device)
        self.navi_former.eval()

    def _load_checkpoint(self, ckpt_path):
        if not ckpt_path:
            raise ValueError("Bridge-DP checkpoint path is empty")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        raw = torch.load(ckpt_path, map_location=self.device)
        if isinstance(raw, dict):
            for sub_key in ('state_dict', 'model_state_dict', 'model'):
                if sub_key in raw and isinstance(raw[sub_key], dict):
                    print(f"[BridgeDP] unwrap checkpoint key '{sub_key}'")
                    raw = raw[sub_key]
                    break

        model_keys = set(self.navi_former.state_dict().keys())

        def try_load(candidate, label):
            overlap = model_keys & set(candidate.keys())
            if len(overlap) <= max(1, int(len(model_keys) * 0.35)):
                return False
            missing, unexpected = self.navi_former.load_state_dict(candidate, strict=False)
            print(
                f"[BridgeDP] loaded checkpoint via {label}: "
                f"loaded={len(model_keys) - len(missing)}/{len(model_keys)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[BridgeDP] missing examples: {missing[:5]}")
            if unexpected:
                print(f"[BridgeDP] unexpected examples: {unexpected[:5]}")
            return True

        if try_load(raw, "direct"):
            return

        for prefix in ('model.', 'module.', 'navi_former.', 'bridgedp.'):
            stripped = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            if stripped and try_load(stripped, f"strip '{prefix}'"):
                return
            added = {prefix + k: v for k, v in raw.items()}
            if try_load(added, f"add '{prefix}'"):
                return

        print("[BridgeDP] automatic key matching failed; forcing non-strict load")
        missing, unexpected = self.navi_former.load_state_dict(raw, strict=False)
        print(
            f"[BridgeDP] forced load: loaded={len(model_keys) - len(missing)}/{len(model_keys)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def reset(self, batch_size, threshold):
        self.batch_size = int(batch_size)
        self.stop_threshold = threshold
        self.memory_queue = [[] for _ in range(self.batch_size)]
        self.prior_curve_queue = [None for _ in range(self.batch_size)]
        self.retry_sigma_scale = np.ones((self.batch_size,), dtype=np.float32)
        self.last_safety_metadata = self._empty_safety_metadata()

    def reset_env(self, i):
        self.memory_queue[i] = []
        self.prior_curve_queue[i] = None
        self.retry_sigma_scale[i] = 1.0

    def _empty_safety_metadata(self):
        return {
            "status": ["disabled" for _ in range(getattr(self, "batch_size", 0))],
            "selected_min_clearance_m": [None for _ in range(getattr(self, "batch_size", 0))],
            "safe_candidate_count": [0 for _ in range(getattr(self, "batch_size", 0))],
            "retry_sigma_scale": [1.0 for _ in range(getattr(self, "batch_size", 0))],
        }

    def process_image(self, images):
        assert len(images.shape) == 4
        H, W = images.shape[1], images.shape[2]
        prop = self.image_size / max(H, W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            return_images.append(resize_image.astype(np.float32) / 255.0)
        return np.array(return_images)

    def process_depth(self, depths):
        assert len(depths.shape) == 4
        depths = depths.copy()
        depths[depths == np.inf] = 0
        H, W = depths.shape[1], depths.shape[2]
        prop = self.image_size / max(H, W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0]) // 2, 0)
            pad_depth = np.pad(
                resize_depth,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant',
                constant_values=0,
            )
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            resize_depth[resize_depth > 5.0] = 0
            resize_depth[resize_depth < 0.1] = 0
            return_depths.append(resize_depth[:, :, np.newaxis])
        return np.array(return_depths)

    def process_pixel(self, pixel_coords, input_images):
        return_pixels = []
        H, W = input_images.shape[1], input_images.shape[2]
        prop = self.image_size / max(H, W)
        for pixel_coord, input_image in zip(pixel_coords, input_images):
            panel_image = np.zeros_like(input_image, dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10

            if min_x <= 0:
                panel_image[:, 0:10] = 255
            elif min_y <= 0:
                panel_image[0:10, :] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:, panel_image.shape[1] - 10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0] - 10:, :] = 255
            else:
                panel_image[min_y:max_y, min_x:max_x] = 255

            resize_image = cv2.resize(panel_image, (-1, -1), fx=prop, fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            return_pixels.append(resize_image.astype(np.float32) / 255.0)
        return np.array(return_pixels).mean(axis=-1)

    def process_pointgoal(self, goals):
        return goals.clip(-10, 10)

    def project_trajectory(self, images, n_trajectories, n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i, :, :, 0:2]
            n_value = n_values[i]
            for waypoints, value in zip(n_trajectory, n_value):
                norm_value = np.clip(-value * 0.1, 0, 1)
                color = cv2.applyColorMap(
                    np.array([[int(norm_value * 255.0)]], dtype=np.uint8),
                    cv2.COLORMAP_JET,
                )[0, 0]
                input_points = np.zeros((waypoints.shape[0], 3)) - 0.2
                input_points[:, 0:2] = waypoints
                input_points[:, 1] = -input_points[:, 1]
                camera_z = (
                    images[0].shape[0] - 1
                    - self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8)
                    - self.image_intrinsic[1][2]
                )
                camera_x = (
                    self.image_intrinsic[0][0] * input_points[:, 1] / (input_points[:, 0] + 1e-8)
                    + self.image_intrinsic[0][2]
                )
                for j in range(camera_x.shape[0] - 1):
                    try:
                        if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j + 1] > 0 and camera_z[j + 1] > 0:
                            trajectory_mask = cv2.line(
                                trajectory_mask,
                                (int(camera_x[j]), int(camera_z[j])),
                                (int(camera_x[j + 1]), int(camera_z[j + 1])),
                                color.tolist(),
                                5,
                            )
                    except Exception:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks, axis=1)

    def _build_input_images(self, images):
        process_images = self.process_image(images)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(
                    input_image,
                    ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0)),
                )
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)
        return np.array(input_images)

    def _get_prior_trajs(self):
        prior_trajs = []
        for i in range(len(self.memory_queue)):
            if self.prior_curve_queue[i] is not None:
                prior_trajs.append(self.prior_curve_queue[i])
            else:
                prior_trajs.append(np.zeros((self.predict_size, 3), dtype=np.float32))
        return np.array(prior_trajs, dtype=np.float32)

    def _update_prior(self, raw_good_trajectory):
        for i in range(len(self.memory_queue)):
            if raw_good_trajectory.ndim == 4:
                self.prior_curve_queue[i] = raw_good_trajectory[i, 0].copy()
            else:
                self.prior_curve_queue[i] = raw_good_trajectory[i].copy()

    def _thresholds(self):
        thresholds = np.asarray(self.stop_threshold, dtype=np.float32)
        if thresholds.ndim == 0:
            thresholds = np.full((self.batch_size,), float(thresholds), dtype=np.float32)
        return thresholds.reshape(-1)

    def _apply_stop_fallback(self, raw_good_trajectory, all_values):
        thresholds = self._thresholds()
        raw_good_trajectory = raw_good_trajectory.copy()
        for i in range(raw_good_trajectory.shape[0]):
            threshold = thresholds[min(i, thresholds.shape[0] - 1)]
            if np.nanmax(all_values[i]) < threshold:
                y_mean = raw_good_trajectory[i, :, :, 1].mean()
                direction = np.sign(y_mean) if abs(y_mean) > 1e-6 else 1.0
                raw_good_trajectory[i, :, :, 0] = 0.0
                raw_good_trajectory[i, :, :, 1] = direction
        return raw_good_trajectory

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _resample_curve_to_exec_waypoints(self, curve):
        curve = np.asarray(curve, dtype=np.float32)
        output = np.zeros((self.exec_num_waypoints, 3), dtype=np.float32)
        if curve.size == 0:
            return output
        if curve.shape[-1] < 3:
            padded = np.zeros((curve.shape[0], 3), dtype=np.float32)
            padded[:, :curve.shape[-1]] = curve
            curve = padded

        origin = np.zeros((1, 3), dtype=np.float32)
        points = np.concatenate([origin, curve[:, :3]], axis=0)
        segment_len = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:] = segment_len > 1e-5
        points = points[keep]

        if points.shape[0] < 2:
            return output

        segment_len = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(segment_len)])
        total_len = float(arc[-1])
        if total_len <= 1e-5:
            return output

        target_arc = self.exec_waypoint_spacing * (np.arange(self.exec_num_waypoints, dtype=np.float32) + 1.0)
        target_arc = np.minimum(target_arc, total_len)
        output[:, 0] = np.interp(target_arc, arc, points[:, 0])
        output[:, 1] = np.interp(target_arc, arc, points[:, 1])
        theta = np.unwrap(points[:, 2])
        output[:, 2] = self._wrap_to_pi(np.interp(target_arc, arc, theta))
        return output

    def _resample_candidates_to_exec(self, trajectories):
        trajectories = np.asarray(trajectories, dtype=np.float32)
        output_shape = trajectories.shape[:-2] + (self.exec_num_waypoints, 3)
        output = np.zeros(output_shape, dtype=np.float32)
        for index in np.ndindex(trajectories.shape[:-2]):
            output[index] = self._resample_curve_to_exec_waypoints(trajectories[index])
        return output

    def _depth_to_obstacle_points(self, depth):
        """Project depth at robot-body height into local forward/lateral points."""
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            return np.zeros((0, 2), dtype=np.float32)
        intrinsic = np.asarray(self.image_intrinsic, dtype=np.float32)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        height, width = depth.shape
        stride = 2
        vv, uu = np.mgrid[0:height:stride, 0:width:stride]
        forward = depth[::stride, ::stride]
        expected_v = (
            height - 1
            - fy * self.safety_projection_height_m / np.maximum(forward, 1e-6)
            - cy
        )
        valid = (
            np.isfinite(forward)
            & (forward > 0.1)
            & (forward < self.safety_depth_max_m)
            & (np.abs(vv - expected_v) <= self.safety_height_band_px)
        )
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float32)
        lateral = -(uu[valid] - cx) * forward[valid] / max(fx, 1e-6)
        return np.stack([forward[valid], lateral], axis=-1).astype(np.float32)

    def _dense_exec_path(self, trajectory):
        points = np.concatenate(
            [np.zeros((1, 2), dtype=np.float32), np.asarray(trajectory, dtype=np.float32)[:, :2]],
            axis=0,
        )
        dense = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(end - start))
            steps = max(1, int(np.ceil(length / self.safety_path_sample_spacing_m)))
            alpha = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:]
            dense.extend(start[None, :] + alpha[:, None] * (end - start)[None, :])
        return np.asarray(dense, dtype=np.float32)

    def _candidate_min_clearances(self, candidates, depth):
        obstacles = self._depth_to_obstacle_points(depth)
        if obstacles.shape[0] == 0:
            return np.full((candidates.shape[0],), np.inf, dtype=np.float32)
        min_clearances = []
        for trajectory in candidates:
            dense = self._dense_exec_path(trajectory)
            clearance = np.linalg.norm(
                dense[:, None, :] - obstacles[None, :, :], axis=-1
            ).min()
            min_clearances.append(float(clearance))
        return np.asarray(min_clearances, dtype=np.float32)

    def _select_safe_execution(self, raw_all_trajectory, exec_all_trajectory, all_values, depths):
        selected_raw = np.zeros(
            (self.batch_size, 1, self.predict_size, 3), dtype=np.float32
        )
        selected_exec = np.zeros(
            (self.batch_size, self.exec_num_waypoints, 3), dtype=np.float32
        )
        status = []
        selected_clearance = []
        safe_count = []
        for bid in range(self.batch_size):
            clearances = self._candidate_min_clearances(exec_all_trajectory[bid], depths[bid])
            safe = clearances >= self.safety_clearance_m
            safe_indices = np.flatnonzero(safe)
            safe_count.append(int(safe_indices.size))
            if safe_indices.size > 0:
                choice = safe_indices[np.argmax(all_values[bid, safe_indices])]
                top_choice = int(np.argmax(all_values[bid]))
                status.append("accepted_top1" if choice == top_choice else "accepted_safe_alternative")
                selected_raw[bid, 0] = raw_all_trajectory[bid, choice]
                selected_exec[bid] = exec_all_trajectory[bid, choice]
                selected_clearance.append(
                    None if not np.isfinite(clearances[choice]) else float(clearances[choice])
                )
                self.retry_sigma_scale[bid] = 1.0
            else:
                status.append("blocked_no_safe_candidate")
                selected_clearance.append(
                    None if clearances.size == 0 else float(np.nanmin(clearances))
                )
                self.retry_sigma_scale[bid] = min(
                    self.max_retry_sigma_scale,
                    self.retry_sigma_scale[bid] * self.retry_sigma_growth,
                )
        self.last_safety_metadata = {
            "status": status,
            "selected_min_clearance_m": selected_clearance,
            "safe_candidate_count": safe_count,
            "retry_sigma_scale": self.retry_sigma_scale.tolist(),
        }
        accepted = np.array([item.startswith("accepted") for item in status], dtype=bool)
        for bid in np.flatnonzero(accepted):
            self.prior_curve_queue[bid] = selected_raw[bid, 0].copy()
        return selected_exec

    def _finalize_step(self, images, depths, raw_all_trajectory, all_values, raw_good_trajectory, apply_safety=False):
        exec_all_trajectory = self._resample_candidates_to_exec(raw_all_trajectory)
        if apply_safety and self.enable_safety_layer:
            exec_good_trajectory = self._select_safe_execution(
                raw_all_trajectory, exec_all_trajectory, all_values, depths
            )
        else:
            self.last_safety_metadata = self._empty_safety_metadata()
            raw_good_trajectory = self._apply_stop_fallback(raw_good_trajectory, all_values)
            self._update_prior(raw_good_trajectory)
            exec_good_trajectory = self._resample_candidates_to_exec(raw_good_trajectory)[:, 0]
        trajectory_mask = self.project_trajectory(images, exec_all_trajectory, all_values)
        return exec_good_trajectory, exec_all_trajectory, all_values, trajectory_mask

    def step_nogoal(self, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        prior_trajs = self._get_prior_trajs()

        raw_all, all_values, raw_good, _ = self.navi_former.predict_nogoal_action(
            input_image,
            input_depth,
            prior_traj=prior_trajs,
            sample_num=self.sample_num,
        )
        return self._finalize_step(images, depths, raw_all, all_values, raw_good)

    def step_pointgoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_pointgoal(goals)
        prior_trajs = self._get_prior_trajs()
        theta_g = np.arctan2(input_goals[:, 1], input_goals[:, 0])

        raw_all, all_values, raw_good, _ = self.navi_former.predict_pointgoal_action(
            input_goals,
            input_image,
            input_depth,
            prior_traj=prior_trajs,
            theta_g=theta_g,
            sample_num=self.sample_num,
            sampling_sigma_scale=self.retry_sigma_scale,
        )
        print(all_values.max(), all_values.min())
        return self._finalize_step(
            images, depths, raw_all, all_values, raw_good, apply_safety=True
        )

    def step_imagegoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_image(goals)
        prior_trajs = self._get_prior_trajs()

        raw_all, all_values, raw_good, _ = self.navi_former.predict_imagegoal_action(
            input_goals,
            input_image,
            input_depth,
            prior_traj=prior_trajs,
            sample_num=self.sample_num,
        )
        print(all_values.max(), all_values.min())
        return self._finalize_step(images, depths, raw_all, all_values, raw_good)

    def step_pixelgoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_pixel(goals, images)
        prior_trajs = self._get_prior_trajs()

        raw_all, all_values, raw_good, _ = self.navi_former.predict_pixelgoal_action(
            input_goals,
            input_image,
            input_depth,
            prior_traj=prior_trajs,
            sample_num=self.sample_num,
        )
        return self._finalize_step(images, depths, raw_all, all_values, raw_good)
