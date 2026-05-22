"""Bridge-DP Agent 封装。

独立实现，包含图像/深度预处理、memory 管理、先验轨迹管理，
以及调用 BridgeDP_Policy 网络的推理接口。

与 NavDP Agent 的核心区别：
1. 新增 prior_queue：维护每个环境的上一帧预测轨迹作为先验
2. step_* 方法额外传递 prior_traj 和 theta_g
3. reset 时清空 prior_queue
"""

import cv2
import numpy as np
import torch
from matplotlib import colormaps as cm

from policy_network import BridgeDP_Policy


class BridgeDP_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 sigma_base=1.0,
                 sigma_goal=0.1,
                 n_prior_tokens=4,
                 navi_model="./bridgedp.ckpt",
                 device='cuda:0'):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size

        self.navi_former = BridgeDP_Policy(
            image_size=image_size,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
            sigma_base=sigma_base,
            sigma_goal=sigma_goal,
            n_prior_tokens=n_prior_tokens,
            device=device,
        )
        self._load_checkpoint(navi_model)
        self.navi_former.to(self.device)
        self.navi_former.eval()

    def _load_checkpoint(self, ckpt_path):
        """智能加载 checkpoint，自动处理 key 前缀不匹配问题。

        支持的 checkpoint 格式：
        1. 直接 state_dict（key 完全匹配）
        2. HuggingFace PreTrainedModel 格式（可能带 'model.' 前缀）
        3. InternNav 训练器保存的格式（可能带 'module.' 前缀）
        4. 带 'state_dict' 或 'model_state_dict' 包装的 checkpoint
        """
        import os
        raw = torch.load(ckpt_path, map_location=self.device)

        # 如果 checkpoint 是字典且包含 state_dict 子键，先解包
        if isinstance(raw, dict):
            for sub_key in ('state_dict', 'model_state_dict', 'model'):
                if sub_key in raw and isinstance(raw[sub_key], dict):
                    print(f"[BridgeDP] 解包 checkpoint 子键 '{sub_key}'")
                    raw = raw[sub_key]
                    break

        model_keys = set(self.navi_former.state_dict().keys())
        ckpt_keys = set(raw.keys())

        # 尝试直接加载
        overlap = model_keys & ckpt_keys
        if len(overlap) > len(model_keys) * 0.5:
            missing, unexpected = self.navi_former.load_state_dict(raw, strict=False)
            print(f"[BridgeDP] 直接加载成功: "
                  f"loaded={len(model_keys) - len(missing)}/{len(model_keys)}, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                print(f"[BridgeDP]   missing 示例: {missing[:5]}")
            if unexpected:
                print(f"[BridgeDP]   unexpected 示例: {unexpected[:5]}")
            return

        # 尝试各种前缀映射
        prefixes_to_try = ['model.', 'module.', 'navi_former.', 'bridgedp.']
        for prefix in prefixes_to_try:
            # 情况 A：checkpoint key 多了前缀 → strip
            if any(k.startswith(prefix) for k in ckpt_keys):
                stripped = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
                overlap_a = model_keys & set(stripped.keys())
                if len(overlap_a) > len(model_keys) * 0.5:
                    missing, unexpected = self.navi_former.load_state_dict(stripped, strict=False)
                    print(f"[BridgeDP] strip '{prefix}' 后加载成功: "
                          f"loaded={len(model_keys) - len(missing)}/{len(model_keys)}, "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")
                    if missing:
                        print(f"[BridgeDP]   missing 示例: {missing[:5]}")
                    return

            # 情况 B：model key 多了前缀 → add prefix to ckpt keys
            added = {prefix + k: v for k, v in raw.items()}
            overlap_b = model_keys & set(added.keys())
            if len(overlap_b) > len(model_keys) * 0.5:
                missing, unexpected = self.navi_former.load_state_dict(added, strict=False)
                print(f"[BridgeDP] add '{prefix}' 后加载成功: "
                      f"loaded={len(model_keys) - len(missing)}/{len(model_keys)}, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
                if missing:
                    print(f"[BridgeDP]   missing 示例: {missing[:5]}")
                return

        # 所有自动匹配都失败 → 打印详细诊断信息并强制加载
        print(f"[BridgeDP] ⚠️ 自动 key 匹配失败!")
        print(f"[BridgeDP]   Model keys ({len(model_keys)}) 示例: {sorted(model_keys)[:5]}")
        print(f"[BridgeDP]   Checkpoint keys ({len(ckpt_keys)}) 示例: {sorted(ckpt_keys)[:5]}")
        print(f"[BridgeDP]   重叠 keys: {len(overlap)}")
        missing, unexpected = self.navi_former.load_state_dict(raw, strict=False)
        print(f"[BridgeDP]   强制加载: loaded={len(model_keys) - len(missing)}/{len(model_keys)}")

    def reset(self, batch_size, threshold):
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for _ in range(batch_size)]
        self.prior_queue = [None for _ in range(batch_size)]

    def reset_env(self, i):
        self.memory_queue[i] = []
        self.prior_queue[i] = None

    # ------------------------------------------------------------------
    # 预处理方法
    # ------------------------------------------------------------------

    def process_image(self, images):
        assert len(images.shape) == 4
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        prop = self.image_size / max(H, W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant', constant_values=0,
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image).astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self, depths):
        assert len(depths.shape) == 4
        depths[depths == np.inf] = 0
        H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
        prop = self.image_size / max(H, W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0]) // 2, 0)
            pad_depth = np.pad(
                resize_depth,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant', constant_values=0,
            )
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            resize_depth[resize_depth > 5.0] = 0
            resize_depth[resize_depth < 0.1] = 0
            return_depths.append(resize_depth[:, :, np.newaxis])
        return np.array(return_depths)

    def process_pixel(self, pixel_coords, input_images):
        return_pixels = []
        H, W, C = input_images.shape[1], input_images.shape[2], input_images.shape[3]
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
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y, min_x:max_x] = 255

            resize_image = cv2.resize(
                panel_image, (-1, -1), fx=prop, fy=prop,
                interpolation=cv2.INTER_NEAREST,
            )
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant', constant_values=0,
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image).astype(np.float32) / 255.0
            return_pixels.append(resize_image)
        return np.array(return_pixels).mean(axis=-1)

    def process_pointgoal(self, goals):
        clip_goals = goals.clip(-10, 10)
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)
        return clip_goals

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def project_trajectory(self, images, n_trajectories, n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i, :, :, 0:2]
            n_value = n_values[i]
            for waypoints, value in zip(n_trajectory, n_value):
                norm_value = np.clip(-value * 0.1, 0, 1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
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
                                color.astype(np.uint8).tolist(), 5,
                            )
                    except Exception:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks, axis=1)

    # ------------------------------------------------------------------
    # Memory 管理（构建输入序列）
    # ------------------------------------------------------------------

    def _build_input_images(self, images):
        """处理图像并构建 memory 序列。"""
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
        """获取当前 batch 的先验轨迹。"""
        prior_trajs = []
        for i in range(len(self.memory_queue)):
            if self.prior_queue[i] is not None:
                prior_trajs.append(self.prior_queue[i])
            else:
                prior_trajs.append(np.zeros((self.predict_size, 3)))
        return np.array(prior_trajs)

    def _update_prior(self, good_trajectory):
        """用本帧最优轨迹更新先验队列。"""
        for i in range(len(self.memory_queue)):
            self.prior_queue[i] = good_trajectory[i, 0].copy()

    # ------------------------------------------------------------------
    # 推理入口
    # ------------------------------------------------------------------

    def step_nogoal(self, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        prior_trajs = self._get_prior_trajs()

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_nogoal_action(
                input_image, input_depth, prior_traj=prior_trajs,
            )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        self._update_prior(good_trajectory)
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask

    def step_pointgoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_pointgoal(goals)
        prior_trajs = self._get_prior_trajs()
        theta_g = np.arctan2(input_goals[:, 1], input_goals[:, 0])

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pointgoal_action(
                input_goals, input_image, input_depth,
                prior_traj=prior_trajs, theta_g=theta_g,
            )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        print(all_values.max(), all_values.min())
        self._update_prior(good_trajectory)
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask

    def step_imagegoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_image(goals)
        prior_trajs = self._get_prior_trajs()

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_imagegoal_action(
                input_goals, input_image, input_depth,
                prior_traj=prior_trajs,
            )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        print(all_values.max(), all_values.min())
        self._update_prior(good_trajectory)
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask

    def step_pixelgoal(self, goals, images, depths):
        input_image = self._build_input_images(images)
        input_depth = self.process_depth(depths)
        input_goals = self.process_pixel(goals, images)
        prior_trajs = self._get_prior_trajs()

        all_trajectory, all_values, good_trajectory, bad_trajectory = \
            self.navi_former.predict_pixelgoal_action(
                input_goals, input_image, input_depth,
                prior_traj=prior_trajs,
            )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        self._update_prior(good_trajectory)
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask
