from typing import Dict, List
import json
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import glob
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from transformers import AutoTokenizer
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from unified_video_action.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from unified_video_action.model.common.rotation_transformer import RotationTransformer
from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import SequenceSampler, get_val_mask
from unified_video_action.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

register_codecs()


def _demo_sort_key(key):
    if key.startswith("demo_"):
        suffix = key.split("_")[-1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, key)


def _get_cache_zarr_path(dataset_path, recursive_hdf5, language_emb_model=None):
    suffix = f"_{language_emb_model}" if language_emb_model is not None else ""
    if recursive_hdf5:
        return os.path.normpath(dataset_path) + suffix + ".zarr.zip"
    return dataset_path + suffix + ".zarr.zip"


def _decode_hdf5_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _get_demo_language(demo, hdf5_path, demo_key):
    ep_meta = _decode_hdf5_attr(demo.attrs.get("ep_meta"))
    if isinstance(ep_meta, dict):
        for key in ("lang", "language", "language_instruction"):
            value = ep_meta.get(key)
            if value:
                return str(value)
    raise RuntimeError(
        f"Missing language instruction for {hdf5_path}:{demo_key}. "
        "Expected demo.attrs['ep_meta'] to contain a 'lang' field."
    )


def _remove_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _get_hdf5_paths(dataset_path, recursive_hdf5):
    if recursive_hdf5 and os.path.isdir(dataset_path):
        hdf5_paths = sorted(
            glob.glob(os.path.join(dataset_path, "**", "*.hdf5"), recursive=True)
        )
        hdf5_paths += sorted(
            glob.glob(os.path.join(dataset_path, "**", "*.h5"), recursive=True)
        )
    elif os.path.isdir(dataset_path):
        raise IsADirectoryError(
            f"{dataset_path} is a directory. Set recursive_hdf5=True for RoboCasa-style "
            "directory datasets, or pass a single HDF5 file."
        )
    else:
        hdf5_paths = [dataset_path]

    if len(hdf5_paths) == 0:
        raise RuntimeError(f"No HDF5 files found under {dataset_path}")
    return hdf5_paths


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        language_emb_model=None,
        data_aug=False,
        normalizer_type=None,
        recursive_hdf5=False,
    ):

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:
            cache_zarr_path = _get_cache_zarr_path(
                dataset_path, recursive_hdf5, language_emb_model=language_emb_model
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            print("Cache path:", cache_zarr_path)

            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            recursive_hdf5=recursive_hdf5,
                            language_emb_model=language_emb_model,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        _remove_if_exists(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                recursive_hdf5=recursive_hdf5,
                language_emb_model=language_emb_model,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

    
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        self.data_aug = data_aug

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = (
                    robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                )
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif (
                key.endswith("qpos")
                or key.endswith("ori")
                or key.endswith("states")
                or key.endswith("width")
            ):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if self.n_obs_steps is None:
                assert np.sum(data[key][T_slice] != data[key]) == 0

            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del data[key]

        for key in self.lowdim_keys:
            if self.n_obs_steps is None:
                assert np.sum(data[key][T_slice] != data[key]) == 0

            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    recursive_hdf5=False,
    language_emb_model=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    hdf5_paths = _get_hdf5_paths(dataset_path, recursive_hdf5)
    file_handles = []
    demos_all = []
    language_all = []
    try:
        tokenizer = None
        seq_max_len = None
        if language_emb_model == "clip":
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            seq_max_len = 30
        elif language_emb_model is not None:
            raise NotImplementedError(
                f"Language model {language_emb_model} not implemented"
            )

        for hdf5_path in hdf5_paths:
            print(f"Loading {hdf5_path}")
            file = h5py.File(hdf5_path, "r")
            file_handles.append(file)
            demos = file["data"]
            demo_keys = sorted(
                [key for key in demos.keys() if "actions" in demos[key]],
                key=_demo_sort_key,
            )
            for demo_key in demo_keys:
                demos_all.append((hdf5_path, demos[demo_key]))
                if language_emb_model is not None:
                    language_all.append(
                        _get_demo_language(demos[demo_key], hdf5_path, demo_key)
                    )

        # count total steps
        if len(demos_all) == 0:
            raise RuntimeError(f"No demos with actions found in {dataset_path}")
        if language_emb_model is not None and len(language_all) != len(demos_all):
            raise RuntimeError("Language metadata count does not match demo count.")

        language_tokens_all = None
        if language_emb_model == "clip":
            language_tokens_all = []
            for language in language_all:
                tokens = tokenizer(
                    language,
                    padding="max_length",
                    max_length=seq_max_len,
                    return_tensors="pt",
                )
                language_tokens_all.append(
                    torch.cat(
                        [tokens.input_ids.unsqueeze(1), tokens.attention_mask.unsqueeze(1)],
                        dim=1,
                    )
                )
        episode_ends = list()
        prev_end = 0
        for _, demo in demos_all:
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array(
            "episode_ends",
            episode_ends,
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            if key == "language":
                continue
            this_data = list()
            this_language_data = list()
            for demo_idx, (_, demo) in enumerate(demos_all):
                this_data.append(demo[data_key][:].astype(np.float32))
                if key == "action" and language_tokens_all is not None:
                    this_language_data.append(
                        language_tokens_all[demo_idx].repeat(
                            this_data[-1].shape[0], 1, 1
                        ).numpy()
                    )
            this_data = np.concatenate(this_data, axis=0)
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
                if language_tokens_all is not None:
                    this_language_data = np.concatenate(this_language_data, axis=0)
                    assert this_language_data.shape == (n_steps, 2, seq_max_len)
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
            )
            if key == "action" and language_tokens_all is not None:
                _ = data_group.array(
                    name="language",
                    data=this_language_data,
                    shape=this_language_data.shape,
                    chunks=this_language_data.shape,
                    compressor=None,
                    dtype=this_language_data.dtype,
                )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    meta_c = tuple(shape_meta["obs"][key]["shape"])[0]
                    first_arr = demos_all[0][1][data_key]
                    h, w, c = first_arr.shape[1:]
                    assert c == meta_c, (
                        f"Image channel mismatch for {key}: HDF5 has {c}, "
                        f"shape_meta has {meta_c}"
                    )
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8,
                    )
                    for episode_idx, (_, demo) in enumerate(demos_all):
                        hdf5_arr = demo["obs"][key]
                        assert hdf5_arr.shape[1:] == (h, w, c), (
                            f"Image shape mismatch for {key}: expected {(h, w, c)}, "
                            f"got {hdf5_arr.shape[1:]}"
                        )
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))
    finally:
        for file in file_handles:
            file.close()

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
