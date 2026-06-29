import argparse
import ast
import json
import pickle
import re
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_robocasa_uva import (  # noqa: E402
    build_video_frame,
    check_success,
    get_env_action_dim,
    merge_registry_task_kwargs,
    safe_name,
    write_video,
)


def parse_demo_key(value):
    if isinstance(value, str) and value.startswith("demo_"):
        return value
    return f"demo_{int(value)}"


def load_controller_configs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def maybe_literal(value):
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def get_attr_json(attrs, key):
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def infer_task_name(hdf5_file, demo):
    ep_meta = get_attr_json(demo.attrs, "ep_meta")
    if isinstance(ep_meta, dict):
        for key in ("env_name", "task_name", "task"):
            if ep_meta.get(key):
                return ep_meta[key]

    env_args = get_attr_json(hdf5_file["data"].attrs, "env_args")
    if isinstance(env_args, dict):
        env_kwargs = env_args.get("env_kwargs") or {}
        return env_kwargs.get("env_name") or env_args.get("env_name")
    return None


def get_initial_state(demo):
    if "states" not in demo or demo["states"].shape[0] == 0:
        return None
    return np.asarray(demo["states"][0])


def set_env_state(env, state):
    if state is None:
        return False
    target = env.env if hasattr(env, "env") else env
    sim = getattr(target, "sim", None)
    if sim is None:
        return False
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(state)
        sim.forward()
        return True
    if hasattr(sim, "set_state"):
        sim.set_state(state)
        sim.forward()
        return True
    return False


def make_env(args, task_name):
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
        "use_object_obs": True,
        "has_renderer": args.render,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "reward_shaping": False,
        "seed": args.seed,
        "obj_instance_split": args.obj_instance_split,
        "layout_and_style_ids": maybe_literal(args.layout_and_style_ids),
        "generative_textures": None,
        "randomize_cameras": args.randomize_cameras,
        "translucent_robot": False,
        "clutter_mode": args.clutter_mode,
        "control_freq": args.control_freq,
    }
    env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    env_kwargs = merge_registry_task_kwargs(task_name, env_kwargs)
    while True:
        try:
            return robosuite.make(**env_kwargs), env_kwargs
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if match is None:
                raise
            bad_key = match.group(1)
            if bad_key not in env_kwargs:
                raise
            print(f"RoboCasa env does not accept '{bad_key}'; retrying without it.")
            env_kwargs.pop(bad_key)


def main():
    parser = argparse.ArgumentParser(
        description="Replay one RoboCasa training demo with its recorded HDF5 actions."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--task", default=None)
    parser.add_argument("--controller_configs_path", default="unified_video_action/config/robocasa_controller_configs.pkl")
    parser.add_argument("--robots", default="PandaMobile")
    parser.add_argument("--seed", type=int, default=1111111)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--env_img_res", type=int, default=224)
    parser.add_argument("--obj_instance_split", default="train")
    parser.add_argument("--layout_and_style_ids", default=None)
    parser.add_argument("--clutter_mode", type=int, default=None)
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default="data/outputs/robocasa_demo_replay/videos")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--video_stride", type=int, default=1)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    demo_key = parse_demo_key(args.demo)
    controller_path = Path(args.controller_configs_path)
    if not controller_path.is_absolute():
        controller_path = REPO_ROOT / controller_path
    args.controller_configs_path = str(controller_path)

    with h5py.File(dataset_path, "r") as hdf5_file:
        demo = hdf5_file[f"data/{demo_key}"]
        actions = np.asarray(demo["actions"][:], dtype=np.float32)
        states0 = get_initial_state(demo)
        task_name = args.task or infer_task_name(hdf5_file, demo)
        if task_name is None:
            raise ValueError("Could not infer task name. Pass --task explicitly.")

    env, env_kwargs = make_env(args, task_name)
    frames = []
    success = False
    steps = 0
    try:
        obs = env.reset()
        state_set = set_env_state(env, states0)
        env_action_dim = get_env_action_dim(env)
        if actions.shape[-1] != env_action_dim:
            raise ValueError(
                f"HDF5 action dim {actions.shape[-1]} does not match env action dim "
                f"{env_action_dim}. Check robot/controller/action layout."
            )

        print(f"task={task_name}")
        print(f"demo={demo_key}")
        print(f"actions_shape={actions.shape}")
        print(f"env_action_dim={env_action_dim}")
        print(f"state_restored={state_set}")
        print(f"control_freq={env_kwargs.get('control_freq')}")
        print(f"robots={env_kwargs.get('robots')}")
        print(f"controller={args.controller_configs_path}")

        max_steps = len(actions) if args.max_steps is None else min(args.max_steps, len(actions))
        if args.save_video:
            frames.append(build_video_frame(obs))
        for step_idx in range(max_steps):
            obs, _, done, info = env.step(actions[step_idx])
            steps += 1
            if args.save_video and steps % args.video_stride == 0:
                frames.append(build_video_frame(obs))
            success = check_success(env, info)
            if success or done:
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    video_path = None
    if args.save_video and frames:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = write_video(
            video_dir / f"{safe_name(task_name)}_{demo_key}_success{int(success)}.mp4",
            frames,
            args.video_fps,
        )

    print(f"steps={steps}")
    print(f"success={int(success)}")
    if video_path is not None:
        print(f"video_path={video_path}")


if __name__ == "__main__":
    main()
