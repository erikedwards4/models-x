"""
GPT-2 Transformer Decoder block MLP (multi-layer perceptron),
which is aka the FFN (feed-fwd network) part of each block.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.fn.gelu_new import gelu_new
from models_x.fn.dropout import dropout
from models_x.nn.linear import Linear
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2DecoderBlockMLP"]


@register_dataclass
@dataclass(frozen=True)
class GPT2DecoderBlockMLP():
    """
    GPT-2 Decoder block (layer).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    c_fc: Linear = field(metadata=metadata)
    c_proj: Linear = field(metadata=metadata)

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
        d_inner = int(kwargs.get('d_inner', cfg.d_inner))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Linear 1 (inner)
        c_fc = Linear(in_features=d_model,
                      out_features=d_inner,
                      bias=True,
                      dtype=dtype)

        # Linear 2 (output projection)
        c_proj = Linear(in_features=d_inner,
                        out_features=d_model,
                        bias=True,
                        dtype=dtype)

        return cls(cfg=cfg, c_fc=c_fc, c_proj=c_proj)

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
        params['c_fc'] = self.c_fc.init_params(key=key1)
        params['c_proj'] = self.c_proj.init_params(key=key2)

        return params

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],                # noqa: F722
                 key: Array | None = None,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        I = d_inner (inner dim, hidden dim)
        """
        # Linear 1
        arr = self.c_fc(params=params['c_fc'],
                        arr=arr)                            # B x T x I

        # GELU
        arr = gelu_new(arr=arr)                             # B x T x I

        # Linear 2
        arr = self.c_proj(params=params['c_proj'],
                          arr=arr)                          # B x T x D

        # Dropout
        if not deterministic:
            p = float(getattr(self.cfg, 'p_drop_res', 0.0))
            arr = dropout(arr=arr, key=key, p=p)            # B x T x D

        return arr                                          # B x T x D
