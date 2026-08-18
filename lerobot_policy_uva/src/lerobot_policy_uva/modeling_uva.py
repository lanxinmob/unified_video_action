"""LeRobot adapter around the installed, original UVA policy implementation."""

from __future__ import annotations

import copy
import importlib.util
from collections import deque
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor

from .configuration_uva import UVAConfig


def _uva_install_paths(config: UVAConfig) -> tuple[Path, Path]:
    """Return the installed UVA package directory and repository root."""

    spec = importlib.util.find_spec("unified_video_action")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(
            "The original `unified_video_action` package is not installed. Run:\n"
            "  cd D:/UVA/unified_video_action\n"
            "  pip install -e .\n"
            "before installing or running lerobot_policy_uva."
        )
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    repository_root = package_dir.parent
    config_dir = (
        Path(config.legacy_config_dir).expanduser().resolve()
        if config.legacy_config_dir
        else package_dir / "config"
    )
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"Original UVA Hydra config directory not found: {config_dir}. "
            "Set --policy.legacy_config_dir explicitly if UVA was not installed editable."
        )
    return config_dir, repository_root


def _resolve_asset(path_value: str | None, repository_root: Path) -> Path | None:
    if path_value in (None, "", "null"):
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repository_root / path).resolve()


def _load_legacy_hydra_config(config: UVAConfig):
    """Compose the selected config through UVA's own Hydra configuration tree."""

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf, open_dict
    except ImportError as exc:
        raise ImportError(
            "Faithful UVA loading requires hydra-core and omegaconf from the original UVA environment."
        ) from exc

    config_dir, repository_root = _uva_install_paths(config)
    OmegaConf.register_new_resolver("eval", eval, replace=True)  # noqa: S307 - matches UVA train.py

    stage_overrides = (
        [
            "model.policy.action_model_params.predict_action=false",
            "model.policy.selected_training_mode=video_model",
        ]
        if config.legacy_stage == "video"
        else [
            "model.policy.action_model_params.predict_action=true",
            "model.policy.selected_training_mode=null",
        ]
    )
    config_name = config.legacy_config_name.removesuffix(".yaml")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        legacy_cfg = compose(
            config_name=config_name,
            overrides=[*stage_overrides, *config.legacy_config_overrides],
        )
    OmegaConf.resolve(legacy_cfg)

    with open_dict(legacy_cfg):
        vae_value = config.legacy_vae_checkpoint
        if vae_value is None:
            vae_value = legacy_cfg.model.policy.vae_model_params.autoencoder_path
        vae_path = _resolve_asset(vae_value, repository_root)
        legacy_cfg.model.policy.vae_model_params.autoencoder_path = (
            str(vae_path) if vae_path is not None else None
        )

        mar_value = config.legacy_mar_checkpoint
        if mar_value is None and config.pretrained_path is None:
            mar_value = legacy_cfg.model.policy.autoregressive_model_params.pretrained_model_path
        mar_path = _resolve_asset(mar_value, repository_root)
        legacy_cfg.model.policy.autoregressive_model_params.pretrained_model_path = (
            str(mar_path) if mar_path is not None else None
        )

    if config.require_legacy_assets:
        if vae_path is None or not vae_path.is_file():
            raise FileNotFoundError(
                "The original UVA KL-VAE checkpoint is required but was not found at "
                f"{vae_path}. Set --policy.legacy_vae_checkpoint to kl16.ckpt."
            )
        if mar_path is not None and not mar_path.is_file():
            raise FileNotFoundError(
                "The configured original UVA MAR/video checkpoint was not found at "
                f"{mar_path}. Set --policy.legacy_mar_checkpoint to the correct checkpoint, "
                "or override model.policy.autoregressive_model_params.pretrained_model_path=null "
                "only when intentionally training from scratch."
            )
    return legacy_cfg


def _instantiate_legacy_policy(config: UVAConfig, legacy_cfg):
    """Instantiate UVA's exact UnifiedVideoActionPolicy through Hydra."""

    try:
        import hydra
        from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy
    except ImportError as exc:
        raise ImportError(
            "Could not import the original UnifiedVideoActionPolicy. Install the UVA repository "
            "and its conda/pip runtime dependencies before running LeRobot."
        ) from exc

    policy_cfg = copy.deepcopy(legacy_cfg.model.policy)
    policy = hydra.utils.instantiate(
        policy_cfg,
        task_name=legacy_cfg.task.name,
        task_modes=legacy_cfg.task.task_modes,
        # LeRobot processors apply the same limits/min-max transformation and
        # undo it after inference. Disabling the legacy normalizer prevents a
        # second normalization without changing MAR/VAE/diffusion inputs.
        normalizer_type="none",
        language_emb_model=legacy_cfg.task.dataset.language_emb_model,
    )
    if not isinstance(policy, UnifiedVideoActionPolicy):
        raise TypeError(f"Expected UnifiedVideoActionPolicy, got {type(policy)!r}.")
    return policy


def _tensor_metric(value: Tensor | float | int) -> float:
    return float(value.detach().mean().cpu()) if isinstance(value, Tensor) else float(value)


class _LegacyBatchAdapter:
    """Map LeRobot feature names to the exact observation dictionaries UVA consumes."""

    def __init__(self, config: UVAConfig, legacy_cfg) -> None:
        self.config = config
        self.legacy_cfg = legacy_cfg
        self.task = config.task_family

    @staticmethod
    def _find_key(
        batch: dict[str, Any],
        explicit: str | None,
        candidates: tuple[str, ...],
        *,
        label: str,
    ) -> str:
        if explicit is not None:
            if explicit not in batch:
                raise KeyError(f"Configured {label} key {explicit!r} is absent from the LeRobot batch.")
            return explicit
        for candidate in candidates:
            if candidate in batch:
                return candidate
        lowered = {key: key.lower() for key in batch}
        for candidate in candidates:
            token = candidate.lower()
            for key, low_key in lowered.items():
                if token in low_key:
                    return key
        raise KeyError(f"Could not infer {label}. Tried {candidates}; available keys: {sorted(batch)}")

    @staticmethod
    def _find_component(batch: dict[str, Any], name: str) -> Tensor | None:
        for key, value in batch.items():
            if isinstance(value, Tensor) and (key == name or key.rsplit(".", 1)[-1] == name):
                return value
        return None

    def _primary_image(self, batch: dict[str, Any]) -> Tensor:
        candidates = {
            "pusht": ("observation.image", "image"),
            "libero": ("agentview", "observation.image", "image"),
            "toolhang": ("sideview", "observation.image", "image"),
            "robocasa": ("robot0_agentview_left", "camera1", "left"),
            "umi": ("camera0", "observation.image", "image"),
        }[self.task]
        key = self._find_key(
            batch,
            self.config.primary_image_key,
            candidates,
            label="primary image",
        )
        return batch[key]

    def _state(self, batch: dict[str, Any], expected_dim: int) -> Tensor:
        key = self.config.state_key
        if key not in batch:
            state_keys = [
                candidate
                for candidate, value in batch.items()
                if isinstance(value, Tensor) and candidate.rsplit(".", 1)[-1] == "state"
            ]
            if len(state_keys) != 1:
                raise KeyError(
                    f"Could not uniquely infer the flat state feature; available keys: {sorted(batch)}"
                )
            key = state_keys[0]
        state = batch[key]
        if not isinstance(state, Tensor) or state.shape[-1] != expected_dim:
            actual = getattr(state, "shape", None)
            raise ValueError(
                f"Faithful UVA {self.task} requires a {expected_dim}D state with the original field "
                f"semantics, but {key!r} has shape {actual}. Do not silently slice or pad benchmark state."
            )
        return state

    def make_observations(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        primary = self._primary_image(batch)
        if self.task == "pusht":
            return {"image": primary, "agent_pos": self._state(batch, 2)}
        if self.task == "libero":
            return {"agentview_rgb": primary}
        if self.task == "toolhang":
            wrist_key = self._find_key(
                batch,
                self.config.wrist_image_key,
                ("robot0_eye_in_hand", "wrist"),
                label="wrist image",
            )
            names = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            parts = [self._find_component(batch, name) for name in names]
            if any(part is None for part in parts):
                state = self._state(batch, 9)
                parts = list(torch.split(state, (3, 4, 2), dim=-1))
            return {
                "sideview_image": primary,
                "robot0_eye_in_hand_image": batch[wrist_key],
                **dict(zip(names, parts, strict=True)),
            }
        if self.task == "robocasa":
            wrist_key = self._find_key(
                batch,
                self.config.wrist_image_key,
                ("robot0_eye_in_hand", "wrist", "camera2"),
                label="wrist image",
            )
            right_key = self._find_key(
                batch,
                self.config.right_image_key,
                ("robot0_agentview_right", "right", "camera3"),
                label="right image",
            )
            names = ("ee_pos", "ee_ori", "gripper_states", "joint_states")
            parts = [self._find_component(batch, name) for name in names]
            if any(part is None for part in parts):
                # The released UVA RoboCasa model used 3+3+2+7 fields. Current
                # LeRobot RoboCasa exposes a different 16D state, so it must fail
                # rather than pretend those representations are interchangeable.
                state = self._state(batch, 15)
                parts = list(torch.split(state, (3, 3, 2, 7), dim=-1))
            return {
                "robot0_agentview_left_rgb": primary,
                "robot0_eye_in_hand_rgb": batch[wrist_key],
                "robot0_agentview_right_rgb": batch[right_key],
                **dict(zip(names, parts, strict=True)),
            }

        names = (
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "robot0_gripper_width",
            "robot0_eef_rot_axis_angle_wrt_start",
        )
        parts = [self._find_component(batch, name) for name in names]
        if any(part is None for part in parts):
            state = self._state(batch, 16)
            parts = list(torch.split(state, (3, 6, 1, 6), dim=-1))
        observations = {"camera0_rgb": primary, **dict(zip(names, parts, strict=True))}
        img_indices = self._find_component(batch, "img_indices")
        if img_indices is not None:
            observations["img_indices"] = img_indices
        return observations

    def make_training_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if ACTION not in batch:
            raise KeyError("LeRobot training batch is missing 'action'.")
        action = batch[ACTION]
        if action.ndim != 3 or action.shape[1] != 32:
            raise ValueError(
                f"Faithful UVA expects action shape [B,32,D], got {tuple(action.shape)}. "
                "Keep UVAConfig.action_delta_indices unchanged."
            )
        observations = self.make_observations(batch)
        primary = next(
            value for value in observations.values() if isinstance(value, Tensor) and value.ndim == 5
        )
        if primary.shape[1] != 32:
            raise ValueError(
                f"Faithful UVA expects 32 observation frames during training, got {primary.shape[1]}."
            )
        legacy_batch: dict[str, Any] = {"obs": observations, "action": action}
        from unified_video_action.utils.data_utils import resize_image

        return resize_image(self.legacy_cfg, legacy_batch)


class UVAPolicy(PreTrainedPolicy):
    """LeRobot policy whose model/loss/sampler are the installed original UVA classes."""

    config_class = UVAConfig
    name = "uva"

    def __init__(self, config: UVAConfig, **_: Any) -> None:
        super().__init__(config)
        config.validate_features()
        self.legacy_cfg = _load_legacy_hydra_config(config)
        self.model = _instantiate_legacy_policy(config, self.legacy_cfg)
        self.batch_adapter = _LegacyBatchAdapter(config, self.legacy_cfg)
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        # This is the exact decay/no-decay split used by UVA's get_optimizer().
        return self.model.add_weight_decay(
            self.model.model,
            weight_decay=self.config.optimizer_weight_decay,
        )

    def reset(self) -> None:
        self._observation_history: dict[str, deque[Tensor]] = {}
        self._action_queue: deque[Tensor] = deque()

    def _add_language_latents(self, legacy_batch: dict[str, Any], batch: dict[str, Any]) -> None:
        if self.model.language_emb_model != "clip":
            return
        if "language_latents" in batch:
            legacy_batch["language_latents"] = batch["language_latents"]
            return
        tasks = batch.get("task", batch.get("observation.language"))
        if tasks is None:
            raise ValueError(
                "This UVA config uses CLIP language conditioning, but the batch has no task text."
            )
        if isinstance(tasks, str):
            tasks = [tasks]
        tokens = self.model.tokenizer(
            list(tasks),
            padding="max_length",
            max_length=self.model.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        from unified_video_action.utils.language_model import extract_text_features

        legacy_batch["language_latents"] = extract_text_features(
            self.model.text_model,
            tokens,
            language_emb_model=self.model.language_emb_model,
        )

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        legacy_batch = self.batch_adapter.make_training_batch(batch)
        self._add_language_latents(legacy_batch, batch)
        loss, (video_loss, action_loss) = self.model.compute_loss(legacy_batch)
        metrics = {
            "video_loss": _tensor_metric(video_loss),
            "action_loss": _tensor_metric(action_loss),
            "legacy_mode": self.model.last_selected_mode,
        }
        return loss, metrics

    def _append_observation_history(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        feature_shapes = {
            key: feature.shape for key, feature in (self.config.input_features or {}).items()
        }
        for key, value in batch.items():
            if key not in feature_shapes or not isinstance(value, Tensor):
                prepared[key] = value
                continue
            feature_rank = len(feature_shapes[key])
            if value.ndim == feature_rank + 2:
                prepared[key] = value[:, -self.config.n_obs_steps :]
                continue
            if value.ndim != feature_rank + 1:
                raise ValueError(f"Unexpected rollout shape for {key!r}: {tuple(value.shape)}")
            history = self._observation_history.setdefault(
                key, deque(maxlen=self.config.n_obs_steps)
            )
            history.append(value)
            frames = list(history)
            frames = [frames[0]] * (self.config.n_obs_steps - len(frames)) + frames
            prepared[key] = torch.stack(frames, dim=1)
        return prepared

    @staticmethod
    def _language_goal(batch: dict[str, Any]) -> Any:
        return batch.get("task", batch.get("observation.language"))

    @torch.no_grad()
    def _predict_from_history(self, prepared: dict[str, Any]) -> Tensor:
        if self.config.legacy_stage != "action":
            raise RuntimeError(
                "A video-stage UVA checkpoint has no DiffActLoss action head. Train/load "
                "--policy.legacy_stage=action before rollout or benchmark evaluation."
            )
        observations = self.batch_adapter.make_observations(prepared)
        result = self.model.predict_action(
            observations,
            language_goal=self._language_goal(prepared),
        )
        actions = result["action_pred"]
        if actions.ndim != 3 or actions.shape[1] != self.config.action_horizon:
            raise RuntimeError(
                f"Original UVA returned unexpected action chunk shape {tuple(actions.shape)}."
            )
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **_: Any) -> Tensor:
        prepared = self._append_observation_history(batch)
        return self._predict_from_history(prepared)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **_: Any) -> Tensor:
        prepared = self._append_observation_history(batch)
        if not self._action_queue:
            chunk = self._predict_from_history(prepared)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
