"""LeRobot policy plugin for Unified Video Action."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Install this project with `pip install -e .` "
        "in a Python 3.12 environment."
    ) from exc

from .configuration_uva import UVAConfig
from .modeling_uva import UVAPolicy
from .processor_uva import make_uva_pre_post_processors

__all__ = [
    "UVAConfig",
    "UVAPolicy",
    "make_uva_pre_post_processors",
]
