"""LeRobot configuration for the original Unified Video Action policy."""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("uva")
@dataclass
class UVAConfig(PreTrainedConfig):
    """LeRobot-facing configuration for the installed UVA implementation.

    ``legacy_config_name`` selects one of the original Hydra configurations in
    ``unified_video_action/config``. The policy adapter does not reimplement
    MAR, the KL-VAE, GaussianDiffusion, DiffLoss, or DiffActLoss.

    Original UVA trains on a 32-step sequence with ``pad_before=1`` and
    ``pad_after=7``. LeRobot's sampled index is mapped to slot 1 of that
    sequence; UVA splits its first/second 16 observations and its own
    ``get_trajectory`` selects the corresponding 16-action target.
    """

    legacy_config_name: str = "uva_pusht"
    legacy_config_dir: str | None = None
    legacy_config_overrides: list[str] = field(default_factory=list)
    legacy_stage: Literal["video", "action"] = "action"
    legacy_vae_checkpoint: str | None = None
    legacy_mar_checkpoint: str | None = None
    require_legacy_assets: bool = True

    n_obs_steps: int = 16
    future_observation_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 8
    # Original non-UMI samplers use horizon=32, pad_before=1 and pad_after=7.
    # Deltas -1..30 reproduce that window, while dropping the final 23 anchor
    # frames keeps the largest virtual start at episode_length-32+7.
    drop_n_last_frames: int = 23

    primary_image_key: str | None = None
    wrist_image_key: str | None = None
    right_image_key: str | None = None
    state_key: str = "observation.state"

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.02
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 16 or self.future_observation_steps != 16:
            raise ValueError(
                "Faithful UVA requires n_obs_steps=16 and future_observation_steps=16 "
                "because its training and sampling code is written for a 32-step window."
            )
        if self.action_horizon != 16:
            raise ValueError("Faithful UVA requires action_horizon=16.")
        if self.drop_n_last_frames != 23:
            raise ValueError(
                "Faithful released UVA 32-step windowing requires drop_n_last_frames=23 "
                "for horizon=32, pad_before=1 and pad_after=7."
            )
        if not 1 <= self.n_action_steps <= self.action_horizon:
            raise ValueError("n_action_steps must be in [1, action_horizon].")
        if self.legacy_stage not in {"video", "action"}:
            raise ValueError("legacy_stage must be 'video' or 'action'.")

        # Original UMI training deliberately uses raw low-dimensional values and
        # actions. Other released UVA task configs use limits/min-max scaling.
        if "umi" in self.legacy_config_name.lower():
            self.normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.IDENTITY,
                "ACTION": NormalizationMode.IDENTITY,
            }

    @property
    def task_family(self) -> str:
        name = self.legacy_config_name.lower()
        for family in ("robocasa", "libero", "toolhang", "pusht", "umi"):
            if family in name:
                return family
        raise ValueError(
            f"Cannot infer a supported UVA task family from legacy_config_name={self.legacy_config_name!r}."
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("UVA requires at least one VISUAL input feature.")
        if self.action_feature is None:
            raise ValueError("UVA requires the standard 'action' output feature.")

        expected_action_dims = {
            "pusht": 2,
            "robocasa": 12,
            "libero": 10,
            "toolhang": 10,
            "umi": 10,
        }
        expected = expected_action_dims[self.task_family]
        actual = self.action_feature.shape[-1]
        if actual != expected:
            raise ValueError(
                f"Original UVA {self.task_family} expects action dimension {expected}, got {actual}."
            )

        minimum_cameras = {"robocasa": 3, "toolhang": 2}.get(self.task_family, 1)
        if len(self.image_features) < minimum_cameras:
            raise ValueError(
                f"Original UVA {self.task_family} requires at least {minimum_cameras} camera features."
            )
        for key, feature in self.image_features.items():
            if len(feature.shape) != 3 or feature.shape[0] != 3:
                raise ValueError(f"Image feature {key!r} must have shape (3,H,W), got {feature.shape}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-1, self.n_obs_steps + self.future_observation_steps - 1))

    @property
    def action_delta_indices(self) -> list[int]:
        # UVA receives the same 32-step action window as the observation window;
        # its original get_trajectory() selects sequence slots 15..30 internally.
        return self.observation_delta_indices.copy()

    @property
    def reward_delta_indices(self) -> None:
        return None
