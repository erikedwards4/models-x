"""
Memory Mosaics Transformer Decoder block attention,
which is multi-head attention (MHA) and
uses scaled dot-product attention (SDPA).

Dropout is not supported in jax.nn.dot_product_attention,
so only use this if you don't need attention dropout
(the residual dropout still works), or use this only
for inference. Future releases may change this, so
leaving the code as is to await.

Also: after jax.jit, this is not faster than the
manual implementation (JIT already fuses ver well).
So, use gpt_decoder_block_attn.py.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.fn.dropout import dropout
from models_x.nn.linear import Linear
from models_x.mems.mems_config import MeMsConfig

__all__ = ["MeMsDecoderBlockSDPA"]


@register_dataclass
@dataclass(frozen=True)
class MeMsDecoderBlockSDPA():
    """
    Memory Mosaics Decoder block (layer) attention part.
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: MeMsConfig = field(metadata=metadata)
    qkv_proj: Linear = field(metadata=metadata)
    out_proj: Linear = field(metadata=metadata)

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
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Linear 1: QKV projection
        qkv_proj = Linear(in_features=d_model,
                          out_features=d_model*3,
                          bias=True,
                          dtype=dtype)

        # Linear 2: output projection
        out_proj = Linear(in_features=d_model,
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
        H = nheads (num attn heads): H * Dh = D
        """
        # Input shape
        nbatch, ntoks, d_model = arr.shape

        # Dropouts
        # Change later when attn dropout becomes supported!
        key1, key2 = jax.random.split(key=key, num=2)
        p = 0.0 if deterministic else \
            float(self.cfg.p_drop_attn)

        # Linear 1: project -> queries, keys, values
        qkv = self.qkv_proj(params=params['qkv_proj'],
                            arr=arr)                        # B x T x D*3

        # Reshape and split
        sh = nbatch, ntoks, 3, self.cfg.nheads, self.cfg.d_head
        qkv = qkv.reshape(sh)                           # B x H x 3 x T x Dh
        qkv = qkv.transpose(2, 0, 1, 3, 4)              # 3 x B x T x H x Dh
        q, k, v = qkv[0], qkv[1], qkv[2]                    # B x T x H x Dh

        # SDPA
        arr = jax.nn.dot_product_attention(query=q,
                                           key=k,
                                           value=v,
                                           mask=None,
                                           scale=self.cfg.scale,
                                           is_causal=True,
                                           implementation="xla",
                                           )                # B x T x H x Dh

        # Reshape
        arr = arr.reshape(nbatch, ntoks, d_model)           # B x T x D

        # Linear 2: Output projection
        arr = self.out_proj(params=params['out_proj'],
                            arr=arr)                        # B x T x D

        # Dropout 2: "residual" dropout
        if not deterministic:
            p = float(self.cfg.p_drop_res)
            arr = dropout(arr=arr, key=key2, p=p)           # B x T x D

        return arr                                          # B x T x D
