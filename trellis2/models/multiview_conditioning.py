"""Model-registry wrappers for training multi-view conditioning modules."""

from ..modules.multiview_conditioning import (
    MultiViewConditioningAdapter,
    MultiViewConditioningConfig,
    MultiViewFeatureFusion,
)

__all__ = [
    "MultiViewConditioningAdapter",
    "MultiViewConditioningConfig",
    "MultiViewFeatureFusion",
]
