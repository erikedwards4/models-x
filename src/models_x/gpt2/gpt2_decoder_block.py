"""
GPT-2 Transformer Decoder block (layer),
which are repeated in the main Decoder layers.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.nn.layer_norm import LayerNorm
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_decoder_block_attn import GPT2DecoderBlockAttn
from models_x.gpt2.gpt2_decoder_block_sdpa import GPT2DecoderBlockSDPA
from models_x.gpt2.gpt2_decoder_block_mlp import GPT2DecoderBlockMLP

__all__ = ["GPT2DecoderBlock"]


@register_dataclass
@dataclass(frozen=True)
class GPT2DecoderBlock():
    """
    GPT-2 Decoder block (layer).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    attn: GPT2DecoderBlockAttn = field(metadata=metadata)
    lnorm1: LayerNorm = field(metadata=metadata)
    mlp: GPT2DecoderBlockMLP = field(metadata=metadata)
    lnorm2: LayerNorm = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: GPT2Config,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Config attributes
        d_model = int(kwargs.get('d_model', cfg.d_model))
        nheads = int(kwargs.get('nheads', cfg.nheads))
        p_drop_attn = float(kwargs.get('p_drop_attn', cfg.p_drop_attn))
        p_drop_res = float(kwargs.get('p_drop_res', cfg.p_drop_res))
        attn_implementation = str(kwargs.get('attn_implementation',
                                             cfg.attn_implementation))
        d_inner = int(kwargs.get('d_model', cfg.d_inner))
        lnorm_eps = float(kwargs.get('lnorm_eps', cfg.lnorm_eps))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Decoder block attention
        attn = GPT2DecoderBlockAttn(d_model=d_model,
                                    nheads=nheads,
                                    p_drop_attn=p_drop_attn,
                                    p_drop_res=p_drop_res,
                                    attn_implementation=attn_implementation,
                                    lnorm_eps=lnorm_eps,
                                    dtype=dtype)

        # Layer norm for the 1st Add & Norm (after the Attn)
        lnorm1 = LayerNorm(normalized_shape=d_model,
                           eps=lnorm_eps,
                           bias=True,
                           dtype=dtype)

        # Decoder block MLP
        mlp = GPT2DecoderBlockMLP(d_model=d_model,
                                  d_inner=d_inner,
                                  p_drop=p_drop_res,
                                  dtype=dtype)

        # Layer norm for the 2nd Add & Norm (after the MLP)
        lnorm2 = LayerNorm(normalized_shape=d_model,
                           eps=lnorm_eps,
                           bias=True,
                           dtype=dtype)

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

        # Decoder block attention
        params['attn'] = self.attn.init_params(key1)

        # Decoder block MLP
        params['mlp'] = self.mlp.init_params(key1)

        # Layer norms
        params['lnorm1'] = self.lnorm1.init_params()
        params['lnorm2'] = self.lnorm2.init_params()

        return params

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],            # noqa: F722
                 key: Array | None = None,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:            # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # Attn
        residual = arr                                  # B x T x D
        arr = self.lnorm1(params=params['lnorm1'],
                          arr=arr)                      # B x T x D
        arr = self.attn(params=params['attn'],
                        arr=arr,
                        key=key)                        # B x T x D
        arr = arr + residual                            # B x T x D

        # MLP
        residual = arr                                  # B x T x D
        arr = self.lnorm2(params=params['lnorm2'],
                          arr=arr)                      # B x T x D
        arr = self.mlp(params=params['mlp'],
                       arr=arr,
                       key=key)                         # B x T x D
        arr = arr + residual                            # B x T x D

        return batch                                    # B x T x D
