"""
GPT-2 Transformer Decoder block attention,
which is multi-head attention (MHA) and
uses scaled dot-product attention (SDPA).
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.fn.softmax import softmax
from models_x.fn.dropout import dropout
from models_x.nn.linear import Linear
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2DecoderBlockAttn"]


@register_dataclass
@dataclass(frozen=True)
class GPT2DecoderBlockAttn():
    """
    GPT-2 Decoder block (layer) attention part.
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    qkv_proj: Linear = field(metadata=metadata)
    out_proj: Linear = field(metadata=metadata)

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
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Linear 1: QKV projection
        qkv_proj = Linear(in_features=d_model,
                          out_features=d_model,
                          bias=True,
                          dtype=dtype)

        # Linear 2: output projection
        out_proj = Linear(in_features=d_inner,
                          out_features=d_model,
                          bias=True,
                          dtype=dtype)

        return cls(cfg=cfg, qkv_proj=qkv_proj, out_proj=out_proj)

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
        params['qkv_proj'] = self.qkv_proj.init_params(key=key1)
        params['out_proj'] = self.out_proj.init_params(key=key2)

        return params

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],                # noqa: F722
                 key: Array,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # Linear 1: QKV projection
        arr = self.qkv_proj(params=params['qkv_proj'],
                            arr=arr)                        # B x T x D*3

        # Softmax
        arr = softmax(arr=arr, axis=-1)                     # B x T x D

        # Dropout 1 ("attention" dropout)
        if not deterministic:
            key1, key2 = jax.random.split(key=key, num=2)
            p = float(self.cfg.p_drop_attn)
            arr = dropout(arr=arr, key=key1, p=p)           # B x T x D

        # Linear 2: Output projection
        arr = self.out_proj(params=params['out_proj'],
                            arr=arr)                        # B x T x D

        # Dropout 2 ("residual" dropout)
        if not deterministic:
            p = float(self.cfg.p_drop_res)
            arr = dropout(arr=arr, key=key2, p=p)           # B x T x D

        return arr                                          # B x T x D
