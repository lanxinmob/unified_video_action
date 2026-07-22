import argparse
import ast
import json
import os
import pickle
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

ALL_TASKS = list(TASK_MAX_STEPS.keys())


def parse_seeds(seed_arg):
    if isinstance(seed_arg, str):
        return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    return [int(seed_arg)]


def parse_optional_seeds(seed_arg):
    if seed_arg is None or str(seed_arg).strip() == "":
        return None
    return set(parse_seeds(seed_arg))


def load_controller_configs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_plain_container(value):
    if is_dataclass(value):
        return to_plain_container(asdict(value))
    if isinstance(value, dict):
        return {k: to_plain_container(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_container(v) for v in value]
    return value


def find_nested_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find_nested_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_nested_key(child, key)
            if result is not None:
                return result
    return None


def merge_registry_task_kwargs(task_name, env_kwargs):
    try:
        from robosuite.environments import ALL_ENVIRONMENTS
    except Exception:
        ALL_ENVIRONMENTS = set()

    if task_name in ALL_ENVIRONMENTS:
        return env_kwargs

    task_registry = {}
    try:
        import robocasa.utils.dataset_registry as dataset_registry
    except Exception as exc:
        print(f"Could not import RoboCasa dataset registry: {exc}")
        return env_kwargs

    for name in dir(dataset_registry):
        if name.endswith("TASK_DATASETS"):
            registry_value = getattr(dataset_registry, name)
            if isinstance(registry_value, dict):
                task_registry.update(to_plain_container(registry_value))

    if len(task_registry) == 0:
        print(
            "RoboCasa dataset registry imported, but no '*TASK_DATASETS' dicts "
            "were found."
        )
        return env_kwargs

    task_spec = task_registry.get(task_name)
    if task_spec is None:
        available = ", ".join(sorted(task_registry.keys())[:50])
        print(
            f"Task alias '{task_name}' not found in RoboCasa dataset registry; "
            f"available registry keys start with: {available}"
        )
        return env_kwargs

    task_spec = to_plain_container(task_spec)
    registry_env_meta = find_nested_key(task_spec, "env_meta") or {}
    registry_env_kwargs = find_nested_key(task_spec, "env_kwargs") or {}
    registry_env_name = (
        find_nested_key(task_spec, "env_name")
        or find_nested_key(registry_env_meta, "env_name")
        or task_name
    )

    resolved = {}
    if isinstance(registry_env_meta, dict):
        resolved.update(registry_env_meta.get("env_kwargs", {}))
    if isinstance(registry_env_kwargs, dict):
        resolved.update(registry_env_kwargs)
    resolved.update(env_kwargs)
    resolved["env_name"] = registry_env_name
    print(f"Resolved RoboCasa task alias '{task_name}' -> env_name='{registry_env_name}'")
    return resolved


def parse_layout_and_style_ids(layout_and_style_ids, episode_idx):
    if not layout_and_style_ids:
        return None

    all_layout_style_ids = ast.literal_eval(layout_and_style_ids)
    if episode_idx is None:
        return all_layout_style_ids

    scene_index = episode_idx // 10
    if scene_index >= len(all_layout_style_ids):
        scene_index = scene_index % len(all_layout_style_ids)
    return (all_layout_style_ids[scene_index],)


def cfg_to_plain(value):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return to_plain_container(value)


def get_cfg_env_kwargs(cfg):
    try:
        env_kwargs = cfg.task.env_runner.get("env_kwargs", None)
    except Exception:
        return {}
    if env_kwargs is None:
        return {}
    env_kwargs = cfg_to_plain(env_kwargs)
    return deepcopy(env_kwargs) if isinstance(env_kwargs, dict) else {}


def get_cfg_env_value(cfg, key, default=None):
    try:
        return cfg.task.env_runner.get(key, default)
    except Exception:
        return default


def create_robocasa_env(cfg, args, task_name, seed, episode_idx):
    import robosuite

    try:
        import robocasa.utils.dataset_registry  # noqa: F401
    except Exception:
        pass

    env_kwargs = get_cfg_env_kwargs(cfg)
    env_kwargs.update({
        "env_name": task_name,
        "controller_configs": load_controller_configs(args.controller_configs_path),
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_widths": args.env_img_res,
        "camera_heights": args.env_img_res,
        "control_freq": args.control_freq,
        "camera_depths": False,
        "use_camera_obs": True,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "seed": seed,
    })
    if args.robots is not None:
        env_kwargs["robots"] = args.robots
    else:
        env_kwargs.setdefault("robots", "PandaMobile")
    if args.obj_instance_split is not None:
        env_kwargs["obj_instance_split"] = args.obj_instance_split
    else:
        env_kwargs.setdefault("obj_instance_split", "test")
    if args.layout_and_style_ids is not None:
        env_kwargs["layout_and_style_ids"] = parse_layout_and_style_ids(
            args.layout_and_style_ids, episode_idx
        )
    else:
        env_kwargs.setdefault("layout_and_style_ids", None)
    if args.randomize_cameras:
        env_kwargs["randomize_cameras"] = True
    else:
        env_kwargs.setdefault("randomize_cameras", False)
    env_kwargs.setdefault("use_object_obs", True)
    env_kwargs.setdefault("reward_shaping", False)
    env_kwargs.setdefault("generative_textures", None)
    env_kwargs.setdefault("translucent_robot", False)
    env_kwargs = merge_registry_task_kwargs(task_name, env_kwargs)
    log_env_kwargs = dict(env_kwargs)
    log_env_kwargs["controller_configs"] = args.controller_configs_path
    return robosuite.make(**env_kwargs), log_env_kwargs


def quat_to_axis_angle(quat):
    """Convert RoboSuite ``robot0_eef_quat`` (XYZW) to axis-angle.

    Matching RoboSuite's own conversion is important because the HDF5
    ``obs/ee_ori`` field is already stored as a 3-D axis-angle vector.
    """
    import robosuite.utils.transform_utils as T

    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(
            f"Expected RoboSuite XYZW quaternion with shape (4,), got {quat.shape}"
        )

    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    # quat2axisangle clips q[3] in place, so pass a private copy.
    quat_xyzw = (quat / norm).copy()
    return T.quat2axisangle(quat_xyzw).astype(np.float32)


def orient_robosuite_image(value):
    # robosuite camera observations are vertically flipped relative to normal RGB images.
    if value.ndim == 3 and value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        return value[:, ::-1, :].copy()
    return value[::-1].copy()


def image_to_chw_float(obs, key):
    value = np.asarray(obs[key])
    if value.ndim != 3:
        raise ValueError(f"Expected HWC image for '{key}', got shape {value.shape}")
    value = orient_robosuite_image(value)
    if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        chw = value
    else:
        chw = np.moveaxis(value, -1, 0)
    chw = chw.astype(np.float32)
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw


def image_to_hwc_uint8(obs, key):
    value = np.asarray(obs[key])
    if value.ndim != 3:
        raise ValueError(f"Expected image for '{key}', got shape {value.shape}")
    value = orient_robosuite_image(value)
    if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        value = np.moveaxis(value, 0, -1)
    if value.dtype != np.uint8:
        value = np.clip(value, 0.0, 1.0)
        value = (value * 255).astype(np.uint8)
    return value


def _add_video_overlay(frame, lines):
    if not lines:
        return frame
    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        text = "\n".join(str(line) for line in lines)
        try:
            bbox = draw.multiline_textbbox((0, 0), text, spacing=2)
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_height = 12 * len(lines)
        draw.rectangle((0, 0, image.width, text_height + 8), fill=(0, 0, 0))
        draw.multiline_text((4, 4), text, fill=(255, 255, 255), spacing=2)
        return np.asarray(image)
    except Exception as exc:
        print(f"Video overlay disabled for this frame: {exc}")
        return frame


def build_video_frame(raw_obs, overlay_lines=None):
    frames = [
        image_to_hwc_uint8(raw_obs, "robot0_agentview_left_image"),
        image_to_hwc_uint8(raw_obs, "robot0_agentview_right_image"),
        image_to_hwc_uint8(raw_obs, "robot0_eye_in_hand_image"),
    ]
    frame = np.concatenate(frames, axis=1)
    return _add_video_overlay(frame, overlay_lines)


def safe_name(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value))


def write_video(path, frames, fps):
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return str(path)


def build_frame_obs(raw_obs):
    return {
        "robot0_agentview_left_rgb": image_to_chw_float(
            raw_obs, "robot0_agentview_left_image"
        ),
        "robot0_agentview_right_rgb": image_to_chw_float(
            raw_obs, "robot0_agentview_right_image"
        ),
        "robot0_eye_in_hand_rgb": image_to_chw_float(
            raw_obs, "robot0_eye_in_hand_image"
        ),
        "ee_pos": np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        "ee_ori": quat_to_axis_angle(raw_obs["robot0_eef_quat"]),
        "gripper_states": np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        "joint_states": np.asarray(raw_obs["robot0_joint_pos"], dtype=np.float32),
    }


def stack_obs_window(obs_window, device):
    import torch

    result = {}
    keys = obs_window[0].keys()
    for key in keys:
        array = np.stack([frame[key] for frame in obs_window], axis=0)
        array = np.expand_dims(array, axis=0)
        result[key] = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    return result


def get_env_action_dim(env):
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "action_spec"):
        spec = env.action_spec
        if isinstance(spec, tuple):
            return int(np.asarray(spec[0]).shape[0])
        return int(np.asarray(spec).shape[0])
    raise AttributeError("Could not infer env action dimension")


def get_action_layout(env):
    robot = env.robots[0]
    controller = getattr(robot, "composite_controller", None)

    result = {
        "env_action_dim": getattr(env, "action_dim", None),
        "robot_class": type(robot).__name__,
        "controller_class": (
            type(controller).__name__ if controller is not None else None
        ),
    }

    if controller is None:
        result["robot_attributes"] = [
            name for name in dir(robot)
            if "controller" in name.lower()
        ]
        return result

    split = getattr(controller, "_action_split_indexes", None)
    if split is not None:
        result["split_indexes"] = {
            str(k): [int(v[0]), int(v[1])]
            for k, v in split.items()
        }

    parts = getattr(controller, "part_controllers", None)
    if parts is not None:
        result["part_controllers"] = {}
        for name, part in parts.items():
            result["part_controllers"][str(name)] = {
                "class": type(part).__name__,
                "control_dim": getattr(part, "control_dim", None),
            }

    config = getattr(robot, "part_controller_config", None)
    if config is not None:
        result["part_controller_config_keys"] = list(config.keys())

    if hasattr(controller, "get_action_info_dict"):
        result["action_info"] = controller.get_action_info_dict()
    elif hasattr(controller, "get_action_info"):
        indices, dimensions = controller.get_action_info()
        result["indices"] = list(indices)
        result["dimensions"] = list(dimensions)

    return result


def zero_action(env):
    dim = get_env_action_dim(env)
    return np.zeros(dim, dtype=np.float32)


def validate_action_dim(action, env_action_dim):
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != env_action_dim:
        raise ValueError(
            f"Predicted action dim {action.shape[-1]} does not match RoboCasa "
            f"env action dim {env_action_dim}. Refusing to pad or truncate because "
            "the training and eval action order must match exactly."
        )
    return action


def load_policy(checkpoint_path, output_dir, device):
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict

    payload = torch.load(
        open(checkpoint_path, "rb"),
        pickle_module=dill,
        map_location="cpu",
    )
    cfg = payload["cfg"]

    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.output_dir = str(output_dir)
        if "pretrained_model_path" in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.pretrained_model_path = None
        if "predict_action" in cfg.model.policy.action_model_params:
            cfg.model.policy.action_model_params.predict_action = True
        cfg.model.policy.use_proprioception = True

    workspace_cls = hydra.utils.get_class(cfg.model._target_)
    workspace = workspace_cls(cfg, output_dir=output_dir)
 
    def expand_legacy_proprio_weights(payload, workspace):
        """
        Convert legacy RoboCasa proprio projection:
            8D = ee_pos(3) + ee_ori(3) + gripper(2)

        into current:
            15D = old 8D + joint_states(7)

        The seven new columns are initialized to zero, so the checkpoint
        initially behaves exactly like the legacy model.
        """

        for state_name in ("model", "ema_model"):
            if state_name not in payload.get("state_dicts", {}):
                continue

            target_module = getattr(workspace, state_name, None)
            if target_module is None:
                continue

            source_state = payload["state_dicts"][state_name]
            target_state = target_module.state_dict()

            for source_key in list(source_state.keys()):
                clean_key = source_key.replace("module.", "")

                if not clean_key.endswith(
                    "model.proprioception_proj_cond.weight"
                ):
                    continue

                if clean_key not in target_state:
                    continue

                old_weight = source_state[source_key]
                target_weight = target_state[clean_key]

                if (
                    old_weight.ndim == 2
                    and target_weight.ndim == 2
                    and old_weight.shape[0] == target_weight.shape[0]
                    and old_weight.shape[1] == 8
                    and target_weight.shape[1] == 15
                ):
                    expanded_weight = torch.zeros(
                        target_weight.shape,
                        dtype=old_weight.dtype,
                        device=old_weight.device,
                    )

                    expanded_weight[:, :8] = old_weight
                    source_state[source_key] = expanded_weight

                    print(
                        f"Expanded legacy {state_name} proprio weight: "
                        f"{tuple(old_weight.shape)} -> "
                        f"{tuple(expanded_weight.shape)}"
                    )


    expand_legacy_proprio_weights(payload, workspace)

    workspace.load_payload(
        payload,
        exclude_keys=("optimizer", "lr_scheduler"),
    )


    use_ema = bool(cfg.training.get("use_ema", False))
    policy = workspace.ema_model if use_ema else workspace.model
    if "robocasa" in str(getattr(policy, "task_name", "")).lower() and getattr(
        policy, "language_emb_model", None
    ) is None:
        raise ValueError(
            "This RoboCasa eval requires a language-conditioned checkpoint. "
            "The loaded checkpoint has language_emb_model=None, so it would ignore "
            "the task prompt."
        )
    policy.to(device)
    policy.eval()
    return policy, cfg


def get_n_obs_steps(cfg, override):
    if override is not None:
        return int(override)
    try:
        return int(cfg.task.env_runner.n_obs_steps)
    except Exception:
        pass
    try:
        return int(cfg.n_obs_steps)
    except Exception:
        return 16


def get_language_goal(env):
    if not hasattr(env, "get_ep_meta"):
        return None
    try:
        ep_meta = env.get_ep_meta()
        if isinstance(ep_meta, dict):
            for key in ("lang", "language", "language_instruction", "task_description"):
                value = ep_meta.get(key)
                if value:
                    return value
    except Exception:
        return None
    return None


def check_success(env, info):
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if isinstance(info, dict):
        return bool(info.get("success", False) or info.get("is_success", False))
    return False


def run_episode(
    policy,
    cfg,
    args,
    task_name,
    base_seed,
    episode_idx,
    device,
    save_video=False,
):
    import torch

    # Collision-free deterministic seed. The previous multiplication made
    # episode 0 use seed 0 for every base seed.
    env_seed = int(
        np.random.SeedSequence([int(base_seed), int(episode_idx)])
        .generate_state(1, dtype=np.uint32)[0]
    )
    env, env_kwargs = create_robocasa_env(cfg, args, task_name, env_seed, episode_idx)
    start_time = time.time()
    video_frames = []
    video_path = None
    action_trace = []

    try:
        raw_obs = env.reset()

        expected = {
            "robot0_eef_pos": (3,),
            "robot0_eef_quat": (4,),
            "robot0_gripper_qpos": (2,),
            "robot0_joint_pos": (7,),
        }

        print("\n===== RoboCasa raw observation schema =====")
        print(sorted(raw_obs.keys()))

        for key, expected_shape in expected.items():
            if key not in raw_obs:
                raise KeyError(
                    f"Missing {key}. Available keys: {sorted(raw_obs.keys())}"
                )

            value = np.asarray(raw_obs[key])
            print(
                f"{key}: shape={value.shape}, "
                f"dtype={value.dtype}, value={value}"
            )
            assert value.shape == expected_shape, (
                f"{key}: expected {expected_shape}, got {value.shape}"
            )

        language_goal = get_language_goal(env)
        if not language_goal:
            raise RuntimeError(
                "RoboCasa environment did not provide a language goal. "
                "The checkpoint is language-conditioned, so evaluating without "
                "the episode instruction would make the rollout invalid."
            )
        print(f"language_goal: {language_goal}")

        action_layout = get_action_layout(env)
        if episode_idx == 0:
            print(f"RoboCasa action layout: {action_layout}")

        dummy = zero_action(env)
        for _ in range(args.num_wait_steps):
            raw_obs, _, _, _ = env.step(dummy)

        initial_eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).copy()
        final_eef_pos = initial_eef_pos.copy()

        if save_video:
            video_frames.append(
                build_video_frame(
                    raw_obs,
                    [
                        f"task={task_name} seed={base_seed} episode={episode_idx}",
                        "step=0 success=0",
                        f"language={language_goal}",
                    ] if not args.no_video_overlay else None,
                )
            )

        if hasattr(policy, "reset"):
            policy.reset()

        n_obs_steps = get_n_obs_steps(cfg, args.n_obs_steps)
        obs_window = deque(maxlen=n_obs_steps)
        first_frame = build_frame_obs(raw_obs)
        for _ in range(n_obs_steps):
            obs_window.append(first_frame)

        env_action_dim = get_env_action_dim(env)
        policy_action_dim = getattr(policy, "action_dim", None)
        if policy_action_dim is not None and int(policy_action_dim) != env_action_dim:
            raise ValueError(
                f"Policy action dim {int(policy_action_dim)} does not match "
                f"RoboCasa env action dim {env_action_dim}. Check that the checkpoint "
                "shape_meta, robot, controller_configs, and action layout are the same "
                "as the demonstrations used for training."
            )
        if args.default_max_steps is None:
            config_max_steps = int(get_cfg_env_value(cfg, "max_steps", 1000))
            max_steps = TASK_MAX_STEPS.get(task_name, config_max_steps)
        else:
            max_steps = args.default_max_steps
        success = False
        num_steps = 0

        with torch.no_grad():
            while num_steps < max_steps:
                obs_dict = stack_obs_window(obs_window, device)
                prediction = policy.predict_action(
                    obs_dict=obs_dict, language_goal=language_goal
                )
                action_chunk = prediction["action"][0].detach().cpu().numpy()
                num_open_loop_steps = args.num_open_loop_steps
                if num_open_loop_steps is None:
                    num_open_loop_steps = int(
                        get_cfg_env_value(cfg, "n_action_steps", len(action_chunk))
                    )
                chunk_len = min(num_open_loop_steps, len(action_chunk))

                for action_idx in range(chunk_len):
                    action = validate_action_dim(action_chunk[action_idx], env_action_dim)
                    action_trace.append(action.copy())
                    raw_obs, _, done, info = env.step(action)
                    num_steps += 1
                    final_eef_pos = np.asarray(
                        raw_obs["robot0_eef_pos"], dtype=np.float64
                    ).copy()
                    success = check_success(env, info)

                    if save_video and (num_steps % args.video_stride == 0):
                        action_text = np.array2string(
                            action,
                            precision=2,
                            suppress_small=True,
                            max_line_width=160,
                        )
                        video_frames.append(
                            build_video_frame(
                                raw_obs,
                                [
                                    f"task={task_name} seed={base_seed} episode={episode_idx}",
                                    f"step={num_steps}/{max_steps} success={int(success)}",
                                    f"action={action_text}",
                                ] if not args.no_video_overlay else None,
                            )
                        )
                    obs_window.append(build_frame_obs(raw_obs))

                    if (
                        args.debug_action_every > 0
                        and num_steps % args.debug_action_every == 0
                    ):
                        print(
                            f"step={num_steps} action={np.array2string(action, precision=3)} "
                            f"eef_pos={final_eef_pos} success={int(success)}"
                        )

                    if success or done or num_steps >= max_steps:
                        break

                if success:
                    break

        action_stats = None
        if action_trace:
            actions = np.stack(action_trace, axis=0)
            action_stats = {
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.std(actions, axis=0).tolist(),
                "min": np.min(actions, axis=0).tolist(),
                "max": np.max(actions, axis=0).tolist(),
                "mean_abs": np.mean(np.abs(actions), axis=0).tolist(),
            }

        if save_video:
            video_dir = (
                Path(args.video_dir)
                if args.video_dir
                else Path(args.output_dir) / "videos"
            )
            video_name = (
                f"{safe_name(task_name)}_seed{base_seed}_ep{episode_idx:03d}_"
                f"success{int(success)}.mp4"
            )
            video_path = write_video(
                video_dir / video_name, video_frames, args.video_fps
            )
            print(f"Saved rollout video: {video_path}")

        eef_delta = final_eef_pos - initial_eef_pos
        return {
            "task": task_name,
            "seed": base_seed,
            "episode_idx": episode_idx,
            "env_seed": env_seed,
            "success": success,
            "num_steps": num_steps,
            "elapsed_sec": time.time() - start_time,
            "language_goal": language_goal,
            "env_kwargs": env_kwargs,
            "action_layout": action_layout,
            "action_stats": action_stats,
            "initial_eef_pos": initial_eef_pos.tolist(),
            "final_eef_pos": final_eef_pos.tolist(),
            "eef_displacement": eef_delta.tolist(),
            "eef_displacement_norm": float(np.linalg.norm(eef_delta)),
            "video_path": video_path,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def summarize_results(results):
    by_task = {}
    by_task_seed = {}
    for item in results:
        task = item["task"]
        seed = item["seed"]
        by_task.setdefault(task, []).append(item)
        by_task_seed.setdefault((task, seed), []).append(item)

    task_seed_summary = []
    for (task, seed), items in sorted(by_task_seed.items()):
        successes = [int(x["success"]) for x in items]
        task_seed_summary.append(
            {
                "task": task,
                "seed": seed,
                "num_trials": len(items),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "successes": successes,
                "avg_steps": float(np.mean([x["num_steps"] for x in items])),
            }
        )

    task_summary = []
    for task, items in sorted(by_task.items()):
        successes = [int(x["success"]) for x in items]
        task_summary.append(
            {
                "task": task,
                "num_trials": len(items),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "avg_steps": float(np.mean([x["num_steps"] for x in items])),
            }
        )

    task_rates = [x["success_rate"] for x in task_summary]
    return {
        "overall_trial_success_rate": float(
            np.mean([int(x["success"]) for x in results])
        )
        if results
        else 0.0,
        "mean_task_success_rate": float(np.mean(task_rates)) if task_rates else 0.0,
        "task_seed_summary": task_seed_summary,
        "task_summary": task_summary,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def build_jobs(tasks, seeds, num_shards, shard_id):
    jobs = []
    for task in tasks:
        for seed in seeds:
            jobs.append((task, seed))
    if num_shards <= 1:
        return jobs
    return [job for idx, job in enumerate(jobs) if idx % num_shards == shard_id]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a UVA RoboCasa checkpoint on the fixed 24-task RoboCasa "
            "benchmark: 50 trials per task and 3 seeds by default."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task names. By default all 24 benchmark tasks are evaluated.",
    )
    parser.add_argument("--seeds", default="195,196,197")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--robots", default=None)
    parser.add_argument("--controller_configs_path", default="robocasa_controller_configs.pkl")
    parser.add_argument("--env_img_res", type=int, default=224)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--layout_and_style_ids", default=None)
    parser.add_argument("--obj_instance_split", default=None)
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--num_wait_steps", type=int, default=10)
    parser.add_argument("--num_open_loop_steps", type=int, default=None)
    parser.add_argument("--n_obs_steps", type=int, default=None)
    parser.add_argument("--default_max_steps", type=int, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--video_stride", type=int, default=1)
    parser.add_argument(
        "--video_seeds",
        default=None,
        help=(
            "Optional comma-separated base seeds to record. "
            "For example, --video_seeds 195 records only that seed."
        ),
    )
    parser.add_argument(
        "--num_videos_per_job",
        type=int,
        default=1,
        help="Maximum recorded episodes for each task/seed job.",
    )
    parser.add_argument("--no_video_overlay", action="store_true")
    parser.add_argument(
        "--debug_action_every",
        type=int,
        default=0,
        help="Print action and EEF position every N environment steps; 0 disables it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    tasks = args.tasks if args.tasks is not None else ALL_TASKS
    unknown_tasks = sorted(set(tasks) - set(ALL_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unknown RoboCasa tasks: {unknown_tasks}")
    seeds = parse_seeds(args.seeds)
    video_seeds = parse_optional_seeds(args.video_seeds)
    jobs = build_jobs(tasks, seeds, args.num_shards, args.shard_id)

    expected_trials = len(tasks) * len(seeds) * args.num_trials_per_task
    shard_trials = len(jobs) * args.num_trials_per_task
    print(f"Tasks: {len(tasks)} | seeds: {seeds} | total planned trials: {expected_trials}")
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(jobs)} jobs, {shard_trials} trials")

    policy, cfg = load_policy(args.checkpoint, output_dir, device)

    results = []
    for task_name, seed in jobs:
        job_successes = []
        print(f"\nRunning task={task_name} seed={seed}")
        for episode_idx in range(args.num_trials_per_task):
            save_this_video = (
                args.save_video
                and episode_idx < args.num_videos_per_job
                and (video_seeds is None or seed in video_seeds)
            )
            result = run_episode(
                policy=policy,
                cfg=cfg,
                args=args,
                task_name=task_name,
                base_seed=seed,
                episode_idx=episode_idx,
                device=device,
                save_video=save_this_video,
            )
            results.append(result)
            job_successes.append(int(result["success"]))
            rate = float(np.mean(job_successes))
            print(
                f"  trial {episode_idx + 1:03d}/{args.num_trials_per_task}: "
                f"success={int(result['success'])} steps={result['num_steps']} "
                f"rate={rate:.3f}"
            )

            partial = {
                "checkpoint": args.checkpoint,
                "tasks": tasks,
                "seeds": seeds,
                "num_trials_per_task": args.num_trials_per_task,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "results": results,
                "summary": summarize_results(results),
            }
            write_json(
                output_dir / f"robocasa_eval_shard{args.shard_id}.json",
                partial,
            )

    final = {
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "seeds": seeds,
        "num_trials_per_task": args.num_trials_per_task,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "results": results,
        "summary": summarize_results(results),
    }
    write_json(output_dir / f"robocasa_eval_shard{args.shard_id}.json", final)
    write_json(output_dir / f"robocasa_eval_shard{args.shard_id}_summary.json", final["summary"])

    print("\nDone.")
    print(json.dumps(final["summary"]["task_summary"], indent=2))
    print(f"Mean task success rate: {final['summary']['mean_task_success_rate']:.4f}")
    print(
        "Overall trial success rate: "
        f"{final['summary']['overall_trial_success_rate']:.4f}"
    )


if __name__ == "__main__":
    main()