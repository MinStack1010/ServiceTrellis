from typing import *
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..modules.multiview_conditioning import (
    MultiViewConditioningAdapter,
    MultiViewConditioningConfig,
    MultiViewFeatureFusion,
)
from ..representations import Mesh, MeshWithVoxel


logger = logging.getLogger(__name__)


class MultiViewCheckpointUnavailable(RuntimeError):
    """Raised when a caller requests multi-view inference without trained weights."""


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
        multi_view_fusion: Optional[MultiViewFeatureFusion] = None,
        multi_view_adapter: Optional[MultiViewConditioningAdapter] = None,
        multi_view_config: Optional[dict] = None,
        multi_view_checkpoint_loaded: bool = False,
        allow_untrained_multi_view: bool = False,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.multi_view_fusion = multi_view_fusion
        self.multi_view_adapter = multi_view_adapter
        self.multi_view_config = multi_view_config
        self.multi_view_checkpoint_loaded = multi_view_checkpoint_loaded
        self.multi_view_checkpoint_path: Optional[str] = None
        self.allow_untrained_multi_view = allow_untrained_multi_view
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        # The published checkpoint has no multi-view adapter. A model package
        # may opt in by describing the adapter and shipping a separate trained
        # checkpoint. Randomly initialized adapters are never treated as ready.
        multi_view_config = args.get('multi_view_conditioning')
        if multi_view_config:
            pipeline.configure_multi_view_conditioning(multi_view_config)
            checkpoint_path = (
                os.environ.get("TRELLIS2_MULTI_VIEW_CHECKPOINT")
                or multi_view_config.get("checkpoint")
            )
            if checkpoint_path:
                pipeline.load_multi_view_checkpoint(checkpoint_path)
        pipeline.allow_untrained_multi_view = (
            os.environ.get("TRELLIS2_ALLOW_UNTRAINED_MULTI_VIEW", "").lower()
            in {"1", "true", "yes"}
        )

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)
            if self.multi_view_fusion is not None:
                self.multi_view_fusion.to(device)
            if self.multi_view_adapter is not None:
                self.multi_view_adapter.to(device)

    def _condition_feature_dim(self) -> int:
        """Return and cross-check the feature size consumed by all flow stages."""
        flow_model_keys = [
            'sparse_structure_flow_model',
            'shape_slat_flow_model_512',
            'shape_slat_flow_model_1024',
            'tex_slat_flow_model_512',
            'tex_slat_flow_model_1024',
        ]
        dimensions = {
            model.cond_channels
            for key in flow_model_keys
            if (model := self.models.get(key)) is not None and hasattr(model, 'cond_channels')
        }
        if not dimensions:
            raise RuntimeError("Unable to determine TRELLIS conditioning feature dimension")
        if len(dimensions) != 1:
            raise RuntimeError(f"TRELLIS flow models disagree on cond_channels: {dimensions}")
        return dimensions.pop()

    def configure_multi_view_conditioning(self, config: Optional[dict] = None) -> None:
        """Create untrained multi-view modules from an explicit architecture config.

        This is a development/training hook. Calling it does *not* enable
        reliable multi-view inference: :meth:`load_multi_view_checkpoint` must
        subsequently load compatible trained fusion and adapter weights.
        """
        config = dict(config or {})
        expected_feature_dim = self._condition_feature_dim()
        feature_dim = config.get('feature_dim', expected_feature_dim)
        if feature_dim != expected_feature_dim:
            raise ValueError(
                "multi-view feature_dim must equal the TRELLIS flow model "
                f"cond_channels ({expected_feature_dim}), got {feature_dim}"
            )
        known_keys = set(MultiViewConditioningConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        fusion_config = MultiViewConditioningConfig(**{
            key: config[key] for key in known_keys if key in config
        })
        self.multi_view_fusion = MultiViewFeatureFusion(fusion_config)
        self.multi_view_adapter = MultiViewConditioningAdapter(
            fusion_config.feature_dim,
            mlp_ratio=config.get('adapter_mlp_ratio', 2.0),
        )
        self.multi_view_config = {
            **fusion_config.to_dict(),
            'adapter_mlp_ratio': config.get('adapter_mlp_ratio', 2.0),
        }
        self.multi_view_checkpoint_loaded = False
        self.multi_view_checkpoint_path = None

    @property
    def multi_view_ready(self) -> bool:
        """Whether trained, separate multi-view weights are available."""
        return bool(
            self.multi_view_fusion is not None
            and self.multi_view_adapter is not None
            and self.multi_view_checkpoint_loaded
        )

    def multi_view_status(self) -> dict:
        """Return an explicit status suitable for health/debug reporting."""
        return {
            'configured': self.multi_view_fusion is not None and self.multi_view_adapter is not None,
            'checkpoint_loaded': self.multi_view_checkpoint_loaded,
            'checkpoint_path': self.multi_view_checkpoint_path,
            'allow_untrained_development_mode': self.allow_untrained_multi_view,
            'config': self.multi_view_config,
        }

    def load_multi_view_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained fusion/adapter checkpoint saved by the training hook.

        The checkpoint format is a ``torch.save`` dictionary containing
        ``multi_view_fusion`` and ``multi_view_adapter`` state dicts, plus an
        optional ``config`` matching :class:`MultiViewConditioningConfig`.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Multi-view checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:  # PyTorch versions before weights_only support.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError("Multi-view checkpoint must be a state-dict dictionary")
        checkpoint_config = checkpoint.get('config')
        if self.multi_view_fusion is None or self.multi_view_adapter is None:
            self.configure_multi_view_conditioning(checkpoint_config)
        elif checkpoint_config and self.multi_view_config != {
            **{key: checkpoint_config[key] for key in MultiViewConditioningConfig.__dataclass_fields__ if key in checkpoint_config},  # type: ignore[attr-defined]
            'adapter_mlp_ratio': checkpoint_config.get('adapter_mlp_ratio', 2.0),
        }:
            raise ValueError("Loaded multi-view checkpoint architecture does not match pipeline config")
        try:
            self.multi_view_fusion.load_state_dict(checkpoint['multi_view_fusion'], strict=True)  # type: ignore[union-attr]
            self.multi_view_adapter.load_state_dict(checkpoint['multi_view_adapter'], strict=True)  # type: ignore[union-attr]
        except KeyError as exc:
            raise ValueError(
                "Multi-view checkpoint must contain 'multi_view_fusion' and "
                "'multi_view_adapter' state dicts"
            ) from exc
        self.multi_view_checkpoint_loaded = True
        self.multi_view_checkpoint_path = checkpoint_path

    def save_multi_view_checkpoint(self, checkpoint_path: str) -> None:
        """Save the trainable multi-view modules without the TRELLIS backbone."""
        if self.multi_view_fusion is None or self.multi_view_adapter is None:
            raise RuntimeError("Multi-view conditioning has not been configured")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        torch.save({
            'config': self.multi_view_config,
            'multi_view_fusion': self.multi_view_fusion.state_dict(),
            'multi_view_adapter': self.multi_view_adapter.state_dict(),
        }, checkpoint_path)

    def get_trainable_multi_view_modules(self) -> dict[str, nn.Module]:
        """Expose only the learned fusion/adapter parameters for fine-tuning."""
        if self.multi_view_fusion is None or self.multi_view_adapter is None:
            raise RuntimeError("Multi-view conditioning has not been configured")
        return {
            'multi_view_fusion': self.multi_view_fusion,
            'multi_view_adapter': self.multi_view_adapter,
        }

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the legacy single-view/batched conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): A batch of image
                prompts. A Python list is interpreted as batch dimension ``B``;
                it is not a multi-view representation and is retained only for
                compatibility with published single-view TRELLIS.2 weights.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def _extract_view_features(self, images: Sequence[Image.Image], resolution: int) -> torch.Tensor:
        """Extract DINO tokens for views of one object as ``[1, V, N, C]``."""
        if not 1 <= len(images) <= 4:
            raise ValueError(f"Expected 1-4 images for one object, got {len(images)}")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("images must be PIL.Image.Image instances")
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        features = self.image_cond_model(list(images))
        if self.low_vram:
            self.image_cond_model.cpu()
        if features.ndim != 3 or features.shape[0] != len(images):
            raise RuntimeError(
                "Image conditioner must return [V, N, C] features for a "
                f"single object's views; got {tuple(features.shape)}"
            )
        view_features = features.unsqueeze(0)
        logger.debug(
            "Multi-view DINO features: input_views=%d shape=%s",
            len(images), tuple(view_features.shape),
        )
        return view_features

    def get_multi_view_cond_from_features(
        self,
        features: torch.Tensor,
        *,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[torch.Tensor] = None,
        include_neg_cond: bool = True,
    ) -> dict:
        """Fuse ``[B, V, N, C]`` features into a single TRELLIS condition.

        This method is intentionally separate from :meth:`get_cond`: the
        latter's list argument means independent batch entries, while this
        method preserves a dedicated view axis and produces exactly one
        condition per object in ``B``.
        """
        if features.ndim != 4:
            raise ValueError(
                "Multi-view features must be [B, V, N, C], not a flattened "
                f"batch; got {tuple(features.shape)}"
            )
        if self.multi_view_fusion is None or self.multi_view_adapter is None:
            raise MultiViewCheckpointUnavailable(
                "This TRELLIS checkpoint has no multi-view conditioning "
                "architecture. Configure it and load a trained multi-view "
                "fusion/adapter checkpoint before requesting multiple views."
            )
        if not self.multi_view_ready and not self.allow_untrained_multi_view:
            raise MultiViewCheckpointUnavailable(
                "A trained multi-view conditioning checkpoint is required for "
                "multi-view reconstruction. The released single-view TRELLIS.2 "
                "weights cannot reliably use a randomly initialized adapter."
            )
        if not self.multi_view_ready:
            logger.warning(
                "Using untrained multi-view conditioning in explicit development mode; "
                "output is not a reliable reconstruction."
            )

        # DINO features can be bf16; attention weights must use the same dtype.
        self.multi_view_fusion.to(device=self.device, dtype=features.dtype)
        self.multi_view_adapter.to(device=self.device, dtype=features.dtype)
        try:
            fused = self.multi_view_fusion(features, view_mask, camera_metadata)
            cond = self.multi_view_adapter(fused)
        finally:
            if self.low_vram:
                self.multi_view_fusion.cpu()
                self.multi_view_adapter.cpu()

        logger.debug(
            "Multi-view fused condition: features=%s view_mask=%s fused=%s cond=%s",
            tuple(features.shape),
            None if view_mask is None else tuple(view_mask.shape),
            tuple(fused.shape), tuple(cond.shape),
        )
        if not include_neg_cond:
            return {'cond': cond}
        return {'cond': cond, 'neg_cond': torch.zeros_like(cond)}

    def get_cond_from_images(
        self,
        images: Sequence[Image.Image],
        resolution: int,
        *,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None,
        include_neg_cond: bool = True,
    ) -> dict:
        """Build one condition for one object represented by 1-4 images.

        A single image deliberately uses the original path byte-for-byte in
        spirit (DINO batch size one and the original conditioning shape), so
        existing pretrained single-image inference remains unchanged.
        """
        if not 1 <= len(images) <= 4:
            raise ValueError(f"Expected 1-4 images for one object, got {len(images)}")
        if len(images) == 1:
            return self.get_cond([images[0]], resolution, include_neg_cond)

        features = self._extract_view_features(images, resolution)
        camera_tensor = None
        if camera_metadata is not None:
            camera_tensor = torch.as_tensor(
                camera_metadata, dtype=features.dtype, device=features.device,
            )
            if camera_tensor.ndim == 2:
                camera_tensor = camera_tensor.unsqueeze(0)
        return self.get_multi_view_cond_from_features(
            features,
            view_mask=view_mask,
            camera_metadata=camera_tensor,
            include_neg_cond=include_neg_cond,
        )

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        logger.debug("TRELLIS sparse condition: %s", tuple(cond['cond'].shape))
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        del noise  # free noise tensor
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()
        del z_s  # free latent — no longer needed
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        del decoded  # free decoded voxels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        logger.debug("TRELLIS shape condition: %s", tuple(cond['cond'].shape))
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        del noise

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        del std, mean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        # Free std/mean tensors — no longer needed
        del std, mean

        # Move slat to CPU before loading shape_slat_decoder to GPU.
        # Both together would exceed 22 GB VRAM on an L4.
        slat_cpu = slat.cpu()
        del slat
        import gc as _gc; _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        # Move slat back to GPU only for the upsample call
        slat = slat_cpu.to(self.device)
        del slat_cpu
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        # Free slat immediately after upsample — hr_coords is all we need
        del slat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        del std, mean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        logger.debug("TRELLIS texture condition: %s", tuple(cond['cond'].shape))
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std
        del std, mean  # free normalization tensors

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        del noise  # free noise tensor

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        del std, mean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Optional[Image.Image] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        *,
        images: Optional[Sequence[Image.Image]] = None,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): Legacy single image prompt. It remains fully
                compatible with published single-view checkpoints.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            images (Sequence[Image.Image]): 1-4 views of one physical object.
                More than one image requires a separately trained multi-view
                fusion/adapter checkpoint; it never means a generation batch.
            view_mask (torch.Tensor): Optional ``[1, V]`` valid-view mask.
            camera_metadata: Optional per-view metadata accepted only by a
                multi-view checkpoint configured for its feature dimension.
        """
        if image is not None and images is not None:
            raise ValueError("Pass either image or images, not both")
        object_images = list(images) if images is not None else ([image] if image is not None else [])
        if not 1 <= len(object_images) <= 4:
            raise ValueError(f"Expected 1-4 images for one object, got {len(object_images)}")
        if len(object_images) > 1 and num_samples != 1:
            raise ValueError(
                "Multi-view reconstruction represents one object and requires "
                "num_samples=1; it does not batch independent generations"
            )
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            object_images = [self.preprocess_image(input_image) for input_image in object_images]
        torch.manual_seed(seed)
        cond_512 = self.get_cond_from_images(
            object_images, 512,
            view_mask=view_mask,
            camera_metadata=camera_metadata,
        )
        cond_1024 = self.get_cond_from_images(
            object_images, 1024,
            view_mask=view_mask,
            camera_metadata=camera_metadata,
        ) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
