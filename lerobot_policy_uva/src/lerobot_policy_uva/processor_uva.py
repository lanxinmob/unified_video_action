"""LeRobot normalization boundary for the original UVA policy."""

from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_uva import UVAConfig


def make_uva_pre_post_processors(
    config: UVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reproduce UVA's data normalization with LeRobotDataset statistics.

    Released non-UMI UVA tasks use limits scaling to ``[-1, 1]`` for state and
    action. LeRobot's ``MIN_MAX`` processor performs that same transform before
    the original policy runs and reverses it after action sampling. UMI uses the
    identity mapping, exactly as its original ``normalizer_type=none`` config.

    The wrapped legacy policy is instantiated with ``normalizer_type=none`` so
    these values are never normalized twice.
    """

    return make_default_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
        normalizer_device=config.device,
    )
