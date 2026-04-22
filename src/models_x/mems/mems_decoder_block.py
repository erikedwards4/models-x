"""
Memory Mosaics Transformer Decoder block (layer),
which are repeated in the main Decoder layers.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.nn.layer_norm import LayerNorm
from models_x.mems.mems_config import MeMsConfig
from models_x.mems.mems_decoder_block_attn import MeMsDecoderBlockAttn
from models_x.mems.mems_decoder_block_sdpa import MeMsDecoderBlockSDPA
from models_x.mems.mems_decoder_block_mlp import MeMsDecoderBlockMLP

__all__ = ["MeMsDecoderBlock"]


@register_dataclass
@dataclass(frozen=True)
class MeMsDecoderBlock():
    """
    Memory Mosaics Decoder block (layer).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: MeMsConfig = field(metadata=metadata)
    lnorm1: LayerNorm = field(metadata=metadata)
    lnorm2: LayerNorm = field(metadata=metadata)
    attn: MeMsDecoderBlockAttn | MeMsDecoderBlockSDPA \
        = field(metadata=metadata)
    mlp: MeMsDecoderBlockMLP = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: MeMsConfig,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Config attributes
        d_model = int(kwargs.get('d_model', cfg.d_model))
        attn_implementation = str(kwargs.get('attn_implementation',
                                             cfg.attn_implementation))
        lnorm_eps = float(kwargs.get('lnorm_eps', cfg.lnorm_eps))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Layer norm for the 1st Add & Norm (before the Attn)
        lnorm1 = LayerNorm(normalized_shape=d_model,
                           eps=lnorm_eps,
                           bias=True,
                           dtype=dtype)

        # Layer norm for the 2nd Add & Norm (before the MLP)
        lnorm2 = LayerNorm(normalized_shape=d_model,
                           eps=lnorm_eps,
                           bias=True,
                           dtype=dtype)

        # Decoder block attention
        attn = MeMsDecoderBlockSDPA.from_config(cfg=cfg, **kwargs) \
            if attn_implementation == "sdpa" else \
            MeMsDecoderBlockAttn.from_config(cfg=cfg, **kwargs)

        # Decoder block MLP
        mlp = MeMsDecoderBlockMLP.from_config(cfg=cfg, **kwargs)

        return cls(cfg=cfg, lnorm1=lnorm1, lnorm2=lnorm2, attn=attn, mlp=mlp)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Params dict
        params: dict[str, Any] = {}
        params['lnorm1'] = self.lnorm1.init_params()
        params['lnorm2'] = self.lnorm2.init_params()
        params['attn'] = self.attn.init_params(key1)
        params['mlp'] = self.mlp.init_params(key2)

        return params

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],            # noqa: F722
                 key: Array,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:            # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Attn
        residual = arr                                  # B x T x D
        arr = self.lnorm1(params=params['lnorm1'],
                          arr=arr)                      # B x T x D
        arr = self.attn(params=params['attn'],
                        arr=arr,
                        key=key1)                       # B x T x D
        arr = arr + residual                            # B x T x D

        # MLP
        residual = arr                                  # B x T x D
        arr = self.lnorm2(params=params['lnorm2'],
                          arr=arr)                      # B x T x D
        arr = self.mlp(params=params['mlp'],
                       arr=arr,
                       key=key2)                        # B x T x D
        arr = arr + residual                            # B x T x D

        return arr                                      # B x T x D
