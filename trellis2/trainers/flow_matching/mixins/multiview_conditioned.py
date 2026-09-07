"""Training hook for learned multi-view image conditioning.

The input ``cond`` is a padded image tensor ``[B, V, 3, H, W]`` and
``view_mask`` is ``[B, V]``. Unlike the legacy MultiImageConditionedMixin,
this path never flattens views into independent samples: DINO's temporary
``B*V`` execution batch is immediately restored to ``[B, V, N, C]`` before
the trainable fusion module runs.
"""

from typing import Optional

import torch

from ....modules import image_feature_extractor
from ....utils import dist_utils


class MultiViewConditionedMixin:
    """Mix in learned fusion/adapter conditioning to a flow-matching trainer.

    Training configs must include ``multi_view_fusion`` and
    ``multi_view_adapter`` in their ``models`` section. These are intentionally
    optimized alongside a selected TRELLIS denoiser: the existing single-view
    backbone must normally be fine-tuned as well because its cross-attention
    distribution changes from DINO tokens to fused object tokens.
    """

    def __init__(
        self,
        *args,
        image_cond_model: dict,
        multi_view_fusion_key: str = "multi_view_fusion",
        multi_view_adapter_key: str = "multi_view_adapter",
        freeze_image_cond_model: bool = True,
        **kwargs,
    ):
        self.image_cond_model_config = image_cond_model
        self.multi_view_fusion_key = multi_view_fusion_key
        self.multi_view_adapter_key = multi_view_adapter_key
        self.freeze_image_cond_model = freeze_image_cond_model
        self.image_cond_model = None
        super().__init__(*args, **kwargs)
        if multi_view_fusion_key not in self.models or multi_view_adapter_key not in self.models:
            raise KeyError(
                "Multi-view training requires model entries "
                f"'{multi_view_fusion_key}' and '{multi_view_adapter_key}'"
            )

    def _init_image_cond_model(self):
        with dist_utils.local_master_first():
            self.image_cond_model = getattr(
                image_feature_extractor, self.image_cond_model_config['name']
            )(**self.image_cond_model_config.get('args', {})).cuda()
            self.image_cond_model.model.requires_grad_(not self.freeze_image_cond_model)
            self.image_cond_model.model.train(not self.freeze_image_cond_model)

    def encode_multi_view_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images while restoring the object/view dimensions exactly."""
        if images.ndim != 5:
            raise ValueError(
                "Multi-view training images must be [B, V, 3, H, W], got "
                f"{tuple(images.shape)}"
            )
        if self.image_cond_model is None:
            self._init_image_cond_model()
        batch_size, num_views = images.shape[:2]
        flat_images = images.flatten(0, 1)
        if self.freeze_image_cond_model:
            with torch.no_grad():
                flat_features = self.image_cond_model(flat_images)
        else:
            flat_features = self.image_cond_model(flat_images)
        return flat_features.reshape(batch_size, num_views, *flat_features.shape[1:])

    def _multi_view_models(self):
        # Use DDP-wrapped models when the base trainer enabled DDP.
        model_dict = getattr(self, 'training_models', self.models)
        return (
            model_dict[self.multi_view_fusion_key],
            model_dict[self.multi_view_adapter_key],
        )

    def get_cond(
        self,
        cond: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        features = self.encode_multi_view_images(cond)
        fusion, adapter = self._multi_view_models()
        fused = fusion(features, view_mask=view_mask, camera_metadata=camera_metadata)
        cond = adapter(fused)
        return super().get_cond(cond, neg_cond=torch.zeros_like(cond), **kwargs)

    def get_inference_cond(
        self,
        cond: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        features = self.encode_multi_view_images(cond)
        fusion, adapter = self._multi_view_models()
        fused = fusion(features, view_mask=view_mask, camera_metadata=camera_metadata)
        cond = adapter(fused)
        return super().get_inference_cond(cond, neg_cond=torch.zeros_like(cond), **kwargs)
