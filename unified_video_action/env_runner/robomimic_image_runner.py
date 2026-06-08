import os
import copy
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from unified_video_action.gym_util.async_vector_env import AsyncVectorEnv
from unified_video_action.gym_util.sync_vector_env import SyncVectorEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper
from unified_video_action.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)
from unified_video_action.model.common.rotation_transformer import RotationTransformer

from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.env_runner.base_image_runner import BaseImageRunner
from unified_video_action.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
import gymnasium as gym
from omegaconf import OmegaConf
import robocasa
import robocasa.utils.lerobot_utils as LU


def create_env(split, env_name, seed=None):
    env = gym.make(
        f"robocasa/{env_name}",
        split=split,
        seed=seed
    )
    return env


def _to_plain_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _is_language_obs(key, value):
    key_lower = key.lower()
    obs_type = str(value.get("type", "")).lower()
    return "lang_emb" in key_lower or obs_type in {"language", "lang", "text"}


def _get_language_obs_meta(shape_meta):
    for key, value in shape_meta.get("obs", {}).items():
        if _is_language_obs(key, value):
            return key, tuple(value["shape"])
    return None, None


def _get_env_shape_meta(shape_meta):
    env_shape_meta = copy.deepcopy(shape_meta)
    env_shape_meta["obs"] = {
        key: value
        for key, value in env_shape_meta.get("obs", {}).items()
        if not _is_language_obs(key, value)
    }
    return env_shape_meta


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        env_kwargs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        self.shape_meta = _to_plain_container(shape_meta)
        self.env_shape_meta = _get_env_shape_meta(self.shape_meta)
        self.language_obs_key, self.language_obs_shape = _get_language_obs_meta(
            self.shape_meta
        )
        self._lang_encoder = None
        self._language_embedding_cache = {}

        self.env_kwargs = _to_plain_container(env_kwargs) if env_kwargs is not None else {}
        env_name = self.env_kwargs["env_name"]
        base_seed = self.env_kwargs.get("seed", None)
        if base_seed is not None:
            base_seed = int(base_seed)

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def make_env_fn(env_i):
            def env_fn():
                env_seed = None if base_seed is None else base_seed + env_i
                robocasa_env = create_env(
                    split=self.env_kwargs["split"],
                    env_name=self.env_kwargs["env_name"],
                    seed=env_seed,
                )
                return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicImageWrapper(
                            env=robocasa_env,
                            shape_meta=self.env_shape_meta,
                            init_state=None,
                            render_obs_key=render_obs_key,
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec="h264",
                            input_pix_fmt="rgb24",
                            crf=crf,
                            thread_type="FRAME",
                            thread_count=1,
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render,
                    ),
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps,
                )

            return env_fn

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robocasa_env = create_env(
                split=self.env_kwargs["split"], 
                env_name=env_name,
                seed=base_seed
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robocasa_env,
                        shape_meta=self.env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [make_env_fn(i) for i in range(n_envs)]
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        #with h5py.File(dataset_path, "r") as f:
        """
        for i in range(n_train):
            train_idx = train_start_idx + i
            enable_render = i < n_train_vis
            
            def init_fn(env, seed=train_idx, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to init_state reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None

            env_seeds.append(train_idx)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))
        """
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def _get_language_goals(self, env):
        if self.language_obs_key is None:
            return None

        def get_language_goal_fn(wrapped_env):
            assert isinstance(wrapped_env.env, VideoRecordingWrapper)
            assert isinstance(wrapped_env.env.env, RobomimicImageWrapper)
            return wrapped_env.env.env.get_language_goal()

        lang_goals = env.call_each(
            "run_dill_function",
            args_list=[(dill.dumps(get_language_goal_fn),)] * len(self.env_fns),
        )
        missing = [idx for idx, goal in enumerate(lang_goals) if not goal]
        if missing:
            raise RuntimeError(
                "RoboCasa policy expects language observation "
                f"{self.language_obs_key}, but env.get_ep_meta() did not "
                f"provide a language instruction for env indices {missing}."
            )
        return list(lang_goals)

    def _encode_language_goals(self, language_goals, device):
        missing_goals = [
            goal
            for goal in dict.fromkeys(language_goals)
            if goal not in self._language_embedding_cache
        ]

        if missing_goals:
            if self._lang_encoder is None:
                self._lang_encoder = LangUtils.LangEncoder(device=device)
            with torch.no_grad():
                embeddings = self._lang_encoder.get_lang_emb(missing_goals)
            if torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()
            embeddings = np.asarray(embeddings, dtype=np.float32)

            expected_dim = int(np.prod(self.language_obs_shape))
            if embeddings.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"Language embedding dim mismatch: got {embeddings.shape[-1]}, "
                    f"expected {expected_dim} from shape_meta[{self.language_obs_key}]."
                )

            for goal, embedding in zip(missing_goals, embeddings):
                self._language_embedding_cache[goal] = embedding.reshape(
                    self.language_obs_shape
                )

        return np.stack(
            [self._language_embedding_cache[goal] for goal in language_goals],
            axis=0,
        ).astype(np.float32)

    def _add_language_obs(self, np_obs_dict, language_goals, device):
        if self.language_obs_key is None or language_goals is None:
            return

        first_obs = next(iter(np_obs_dict.values()))
        batch_size, n_obs_steps = first_obs.shape[:2]
        if len(language_goals) != batch_size:
            raise RuntimeError(
                f"Got {len(language_goals)} language goals for batch size {batch_size}."
            )

        lang_emb = self._encode_language_goals(language_goals, device)
        np_obs_dict[self.language_obs_key] = np.repeat(
            lang_emb[:, None, ...], n_obs_steps, axis=1
        )

    def run(self, policy: BaseImagePolicy, **kwargs):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            # past_action = None
            past_action_list = []
            language_goals = self._get_language_goals(env)
            policy.reset()

            env_name = self.env_kwargs["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.language_obs_key not in np_obs_dict:
                    self._add_language_obs(np_obs_dict, language_goals, device)

                if self.past_action:
                    if len(past_action_list) > 1:  ## get 16 actions
                        np_obs_dict["past_action"] = np.concatenate(
                            past_action_list, axis=1
                        )

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, **kwargs)

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                # done = np.all(done)
                # for robocasa switch to the proper success check
                done = np.all(done) or np.all([this_info["success"][0] for this_info in info])

                # past_action = action
                past_action_list.append(action)
                if len(past_action_list) > 2:
                    past_action_list.pop(0)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        success_rate = sum([np.max(all_rewards[i]) > 0 for i in range(n_inits)]) / n_inits
        print(f"Success rate: {success_rate}")
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video
        
        env_name = self.env_kwargs["env_name"]
        log_data[f'success_rate/{env_name}'] = success_rate
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    
    def close(self):
        if not isinstance(self.env, SyncVectorEnv):
            if hasattr(self.env, "close"):
                self.env.close()
            return
        
        # only for SyncVectorEnv
        env_list = self.env.envs
        for env in env_list:
            env_chain = [env]
            while True:
                if hasattr(env, "env"):
                    env = env.env
                else:
                    break
                env_chain = [env] + env_chain
            
            for env in env_chain:
                if hasattr(env, "close"):
                    env.close()
                del env
