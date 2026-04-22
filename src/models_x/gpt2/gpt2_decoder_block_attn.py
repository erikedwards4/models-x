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
        """
        # Input shape
        nbatch, ntoks, d_model = arr.shape

        # Linear 1: project -> queries, keys, values
        qkv = self.qkv_proj(params=params['qkv_proj'],
                            arr=arr)                        # B x T x D*3

        # Reshape for SDPA
        sh = nbatch, ntoks, 3, self.cfg.nheads, self.cfg.d_head
        qkv = qkv.reshape(sh).transpose(0, 3, 2, 1, 4)  # B x H x 3 x T x Dh
        q = qkv[:, :, 0]                                    # B x H x T x Dh
        k = qkv[:, :, 1]                                    # B x H x T x Dh
        v = qkv[:, :, 2]                                    # B x H x T x Dh

        # SDPA scores
        # attn_scores = \
        #     jnp.einsum('bhtd,bhtd->bhtt', q, k) \
        #     * self.cfg.scale                              # B x H x T x T
        kt = k.swapaxes(2, 3)                               # B x H x Dh x T
        attn_scores = jnp.matmul(q, kt) * self.cfg.scale    # B x H x T x T

        # Causal mask
        causal_mask = jnp.ones(shape=(1, 1, ntoks, ntoks),
                               dtype=jnp.bool_)             # 1 x 1 x T x T
        causal_mask = jnp.tril(m=causal_mask, k=0)          # 1 x 1 x T x T

        # Apply mask and softmax
        attn_weights = jax.nn.softmax(x=attn_scores,
                                      axis=-1,
                                      where=causal_mask)    # B x H x T x T

        # Dropout 1 ("attention" dropout)
        if not deterministic:
            key1, key2 = jax.random.split(key=key, num=2)
            p = float(self.cfg.p_drop_attn)
            attn_weights = \
                dropout(arr=attn_weights, key=key1, p=p)    # B x H x T x T

        # Attn-weighted sum of values and reshape
        # arr = jnp.matmul(attn_weights, v)                 # B x H x T x Dh
        # arr = arr.transpose(0, 2, 1, 3)                   # B x T x H x Dh
        arr = jnp.einsum('bhtt,bhtd->bthd',
                         attn_weights,
                         v)                                 # B x T x H x Dh
        arr = arr.reshape(nbatch, ntoks, d_model)           # B x T x D

        # Linear 2: Output projection
        arr = self.out_proj(params=params['out_proj'],
                            arr=arr)                        # B x T x D

        # Dropout 2 ("residual" dropout)
        if not deterministic:
            p = float(self.cfg.p_drop_res)
            arr = dropout(arr=arr, key=key2, p=p)           # B x T x D

        return arr                                          # B x T x D
