from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key="agentview_image",
    ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            obs_type = value.get("type", "low_dim")
            if obs_type == "rgb":
                min_value, max_value = 0, 1
            else:
                min_value, max_value = -1, 1

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    @staticmethod
    def _raw_obs_key(key):
        if key.endswith("_rgb"):
            return key[: -len("_rgb")] + "_image"
        robocasa_key_map = {
            "ee_pos": "robot0_eef_pos",
            "ee_ori": "robot0_eef_quat",
            "gripper_states": "robot0_gripper_qpos",
        }
        return robocasa_key_map.get(key, key)

    @staticmethod
    def _quat_to_axis_angle(quat):
        quat = np.asarray(quat, dtype=np.float32)
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        w = np.clip(quat[..., 0], -1.0, 1.0)
        xyz = quat[..., 1:]
        sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(sin_half[..., 0], w)
        axis = xyz / np.maximum(sin_half, 1e-8)
        return axis * angle[..., None]

    @staticmethod
    def _format_obs_value(key, value, target_shape):
        value = np.asarray(value)
        if key.endswith("_rgb"):
            if value.ndim == 3 and value.shape[-1] == 3:
                value = np.moveaxis(value, -1, 0)
            if value.dtype == np.uint8:
                value = value.astype(np.float32) / 255.0
            else:
                value = value.astype(np.float32)
        elif key == "ee_ori" and value.shape[-1] == 4 and target_shape[-1] == 3:
            value = RobomimicImageWrapper._quat_to_axis_angle(value)
        else:
            value = value.astype(np.float32)
        return value

    @staticmethod
    def _format_render_image(value):
        value = np.asarray(value)
        if value.ndim == 3 and value.shape[-1] == 3:
            value = np.moveaxis(value, -1, 0)
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / 255.0
        return value

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = self._format_render_image(raw_obs[self.render_obs_key])

        obs = dict()
        for key, space in self.observation_space.items():
            raw_key = self._raw_obs_key(key)
            obs[key] = self._format_obs_value(key, raw_obs[raw_key], space.shape)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not hasattr(self.env, "reset_to"):
                raw_obs = self.env.reset()
                return self.get_observation(raw_obs)
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            can_cache_seed = hasattr(self.env, "reset_to") and hasattr(self.env, "get_state")
            if can_cache_seed and seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                if can_cache_seed:
                    state = self.env.get_state()["states"]
                    self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


