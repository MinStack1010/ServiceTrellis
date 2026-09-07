# TRELLIS.2 multi-view conditioning architecture

The released TRELLIS.2 image pipeline is single-view. Its DINO extractor turns
a Python list of images into a batch `[B, N, 1024]`; it does not create a view
dimension or fuse information. `Trellis2ImageTo3DPipeline.get_cond()` retains
that legacy batching behavior for backwards compatibility.

The multi-view path is deliberately separate:

```text
1–4 preprocessed views of one object
        ↓ DINO (temporary execution batch V)
[1, V, N, 1024]
        ↓ learned view embeddings + masked cross-view attention
[1, 256, 1024]
        ↓ learned MultiViewConditioningAdapter
[1, 256, 1024] TRELLIS condition
        ↓ once each: sparse structure → shape SLAT → texture SLAT
one mesh
```

`V` is never used as the generation batch dimension. During fine-tuning,
objects are batched as `[B, V, N, 1024]`; padded input views have a
`[B, V]` `view_mask`, and attention masks every token from an invalid view.
Camera metadata is optional and only used by checkpoints configured with a
known metadata width. Image-only requests do not imply that camera poses are
known.

## Checkpoints and training

The public `microsoft/TRELLIS.2-4B` weights do not include the learned
`MultiViewFeatureFusion` or `MultiViewConditioningAdapter` parameters. A
multi-view checkpoint must contain both state dictionaries and its architecture
config. Load it with `TRELLIS2_MULTI_VIEW_CHECKPOINT=/path/to/checkpoint.pt`.
Without it, the API rejects 2–4-image requests with `503` rather than claiming
that batched DINO features are a reconstruction.

`TRELLIS2_ALLOW_UNTRAINED_MULTI_VIEW=1` is an explicit development-only mode
for exercising tensor plumbing. It must not be used to assess reconstruction
quality or exposed as a production capability.

Fine-tune the fusion module and adapter with the desired sparse-structure,
shape-SLAT, and texture-SLAT flow model(s). All three stages consume the image
condition, so adapters trained only for sparse structure are insufficient for
consistent geometry and texture. The `MultiViewConditionedSparseFlowMatchingCFGTrainer`
hook restores DINO's temporary `[B*V, N, 1024]` execution batch to
`[B, V, N, 1024]` before fusion; it is the training replacement for the older
token-flattening experimental mixin.
