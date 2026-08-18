"""FastWAM-D — FastWAM com profundidade métrica. Ver README.md."""

from .configuration_fastwam_depth import FastWAMDepthConfig
from .modeling_fastwam_depth import FastWAMDepthPolicy
from .processor_fastwam_depth import make_fastwamdepth_pre_post_processors

__all__ = [
    "FastWAMDepthConfig",
    "FastWAMDepthPolicy",
    "make_fastwamdepth_pre_post_processors",
]
