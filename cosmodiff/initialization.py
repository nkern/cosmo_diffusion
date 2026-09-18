"""Explicit, fresh-model initialization policies."""


def initialize_dit_adaln_zero(model, *, fresh: bool) -> dict:
    """Zero a fresh native DiT's modulation and final output projections.

    This adopts the zero-modulation/output part of the original DiT recipe,
    not its complete initialization scheme. Attention, FFN, patch embedding,
    and timestep/class embedding parameters retain diffusers' initialization.
    All parameters remain trainable. Never apply this to a loaded checkpoint.

    The projection API is validated in full before any parameter is changed.
    Unsupported architectures fail explicitly rather than receiving a partial
    initialization. The marker prevents applying this policy twice in memory.

    Returns:
        dict: Names and count of the initialized projections, for inspection.
    """
    import torch
    from diffusers import DiTTransformer2DModel
    from diffusers.models.normalization import AdaLayerNormZero

    if not fresh or getattr(model, "_cosmodiff_adaln_zero_initialized", False):
        raise RuntimeError("adaLN-Zero initialization requires a fresh, uninitialized model")
    if type(model) is not DiTTransformer2DModel:
        raise TypeError("adaLN-Zero initialization supports only native DiTTransformer2DModel")

    width = model.config.num_attention_heads * model.config.attention_head_dim
    if len(model.transformer_blocks) != model.config.num_layers:
        raise ValueError("DiT layer count differs from its config")

    targets = []
    for index, block in enumerate(model.transformer_blocks):
        if type(block.norm1) is not AdaLayerNormZero:
            raise TypeError(f"DiT block {index} requires norm_type='ada_norm_zero'")
        targets.append((
            f"transformer_blocks.{index}.norm1.linear", block.norm1.linear, 6 * width,
        ))
    targets.extend([
        ("proj_out_1", model.proj_out_1, 2 * width),
        ("proj_out_2", model.proj_out_2, model.config.patch_size**2 * model.out_channels),
    ])

    for name, layer, outputs in targets:
        if (type(layer) is not torch.nn.Linear or layer.bias is None
                or tuple(layer.weight.shape) != (outputs, width)
                or tuple(layer.bias.shape) != (outputs,)
                or layer.weight.is_meta or layer.bias.is_meta):
            raise ValueError(f"Unsupported DiT projection API: {name}")

    with torch.no_grad():
        for _, layer, _ in targets:
            layer.weight.zero_()
            layer.bias.zero_()
    model._cosmodiff_adaln_zero_initialized = True

    return {
        "kind": "adaln_zero_modulation_and_output",
        "fresh_only": True,
        "linear_modules": [name for name, _, _ in targets],
        "zeroed_tensors": 2 * len(targets),
    }
