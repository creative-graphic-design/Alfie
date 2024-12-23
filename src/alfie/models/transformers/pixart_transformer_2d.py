from diffusers.models import PixArtTransformer2DModel


class AlfinePixArtTransformer2DModel(PixArtTransformer2DModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: int | None = 8,
        num_layers: int = 28,
        dropout: float = 0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 0.000001,
        interpolation_scale: int | None = None,
        use_additional_conditions: bool | None = None,
        caption_channels: int | None = None,
        attention_type: str | None = "default",
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            use_additional_conditions,
            caption_channels,
            attention_type,
        )
