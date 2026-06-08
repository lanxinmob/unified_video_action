import json
from typing import Optional
import numpy as np
import gym
from gym import spaces
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
            if self._is_policy_only_obs(key, value):
                continue

            shape = value["shape"]
            min_value, max_value = -1, 1
            obs_type = value.get("type", "low_dim")
            if obs_type == "rgb" or key.endswith("image"):
                min_value, max_value = 0, 1
            elif obs_type == "low_dim" or key.endswith("quat") or key.endswith("qpos") or key.endswith("pos"):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported observation key {key} with type {obs_type}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    @staticmethod
    def _is_policy_only_obs(key, value):
        key_lower = key.lower()
        obs_type = str(value.get("type", "")).lower()
        return "lang_emb" in key_lower or obs_type in {"language", "lang", "text"}

    @staticmethod
    def _format_language(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(x) for x in value)
        return str(value)

    def get_language_goal(self):
        ep_meta = None
        candidate_envs = [self.env]
        unwrapped_env = getattr(self.env, "unwrapped", None)
        if unwrapped_env is not None and unwrapped_env is not self.env:
            candidate_envs.append(unwrapped_env)

        for env in candidate_envs:
            if hasattr(env, "get_ep_meta"):
                ep_meta = env.get_ep_meta()
                break

            ep_meta = getattr(env, "ep_meta", None)
            if ep_meta is None:
                ep_meta = getattr(env, "_ep_meta", None)
            if ep_meta is not None:
                break

        if isinstance(ep_meta, str):
            try:
                ep_meta = json.loads(ep_meta)
            except json.JSONDecodeError:
                return ep_meta

        if not isinstance(ep_meta, dict):
            return None

        for key in ("lang", "language", "language_instruction", "instruction"):
            if key in ep_meta:
                return self._format_language(ep_meta[key])
        return None

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
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
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.unwrapped.get_state()["states"]
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


