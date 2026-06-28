import argparse
import ast
import json
import os
import pickle
import time
from collections import deque
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


def load_controller_configs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


def create_robocasa_env(args, task_name, seed, episode_idx):
    import robosuite

    try:
        import robocasa.utils.dataset_registry  # noqa: F401
    except Exception:
        pass

    env_kwargs = {
        "env_name": task_name,
        "robots": args.robots,
        "controller_configs": load_controller_configs(args.controller_configs_path),
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_widths": args.env_img_res,
        "camera_heights": args.env_img_res,
        "camera_depths": False,
        "use_camera_obs": True,
        "use_object_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "reward_shaping": True,
        "seed": seed,
        "obj_instance_split": args.obj_instance_split,
        "generative_textures": None,
        "randomize_cameras": args.randomize_cameras,
        "layout_and_style_ids": parse_layout_and_style_ids(
            args.layout_and_style_ids, episode_idx
        ),
        "translucent_robot": False,
    }
    log_env_kwargs = dict(env_kwargs)
    log_env_kwargs["controller_configs"] = args.controller_configs_path
    return robosuite.make(**env_kwargs), log_env_kwargs


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm

    if quat[0] < 0:
        quat = -quat

    sin_theta = np.linalg.norm(quat[1:])
    if sin_theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = quat[1:] / sin_theta
    angle = 2.0 * np.arctan2(sin_theta, quat[0])
    return (axis * angle).astype(np.float32)


def image_to_chw_float(obs, key):
    value = np.asarray(obs[key])
    if value.ndim != 3:
        raise ValueError(f"Expected HWC image for '{key}', got shape {value.shape}")
    if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        chw = value
    else:
        chw = np.moveaxis(value, -1, 0)
    chw = chw.astype(np.float32)
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw


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


def zero_action(env):
    dim = get_env_action_dim(env)
    return np.zeros(dim, dtype=np.float32)


def adjust_action_dim(action, env_action_dim):
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == env_action_dim:
        return action
    if action.shape[-1] < env_action_dim:
        padded = np.zeros(env_action_dim, dtype=np.float32)
        padded[: action.shape[-1]] = action
        return padded
    return action[:env_action_dim]


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
    workspace.load_payload(payload)

    use_ema = bool(cfg.training.get("use_ema", False))
    policy = workspace.ema_model if use_ema else workspace.model
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
            return ep_meta.get("lang")
    except Exception:
        return None
    return None


def check_success(env, info):
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if isinstance(info, dict):
        return bool(info.get("success", False) or info.get("is_success", False))
    return False


def run_episode(policy, cfg, args, task_name, base_seed, episode_idx, device):
    env_seed = int(base_seed * episode_idx * 256)
    env, env_kwargs = create_robocasa_env(args, task_name, env_seed, episode_idx)
    start_time = time.time()

    try:
        raw_obs = env.reset()
        language_goal = get_language_goal(env)

        dummy = zero_action(env)
        for _ in range(args.num_wait_steps):
            raw_obs, _, _, _ = env.step(dummy)

        if hasattr(policy, "reset"):
            policy.reset()

        n_obs_steps = get_n_obs_steps(cfg, args.n_obs_steps)
        obs_window = deque(maxlen=n_obs_steps)
        first_frame = build_frame_obs(raw_obs)
        for _ in range(n_obs_steps):
            obs_window.append(first_frame)

        env_action_dim = get_env_action_dim(env)
        max_steps = TASK_MAX_STEPS.get(task_name, args.default_max_steps)
        success = False
        num_steps = 0

        with torch.no_grad():
            while num_steps < max_steps:
                obs_dict = stack_obs_window(obs_window, device)
                prediction = policy.predict_action(
                    obs_dict=obs_dict, language_goal=language_goal
                )
                action_chunk = prediction["action"][0].detach().cpu().numpy()
                chunk_len = min(args.num_open_loop_steps, len(action_chunk))

                for action_idx in range(chunk_len):
                    action = adjust_action_dim(action_chunk[action_idx], env_action_dim)
                    raw_obs, _, done, info = env.step(action)
                    num_steps += 1
                    obs_window.append(build_frame_obs(raw_obs))

                    success = check_success(env, info)
                    if success or done or num_steps >= max_steps:
                        break

                if success:
                    break

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
    parser.add_argument("--seeds", default="195,196,197")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--robots", default="PandaMobile")
    parser.add_argument("--controller_configs_path", default="robocasa_controller_configs.pkl")
    parser.add_argument("--env_img_res", type=int, default=224)
    parser.add_argument("--layout_and_style_ids", default="((1,1),(2,2),(4,4),(6,9),(7,10))")
    parser.add_argument("--obj_instance_split", default="B")
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--num_wait_steps", type=int, default=10)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--n_obs_steps", type=int, default=None)
    parser.add_argument("--default_max_steps", type=int, default=500)
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
    tasks = ALL_TASKS
    seeds = parse_seeds(args.seeds)
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
            result = run_episode(
                policy=policy,
                cfg=cfg,
                args=args,
                task_name=task_name,
                base_seed=seed,
                episode_idx=episode_idx,
                device=device,
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
