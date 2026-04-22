"""
Memory Mosaics Stem (stages before the main Decoder).

This is just WTE followed by a Dropout,
noting that WPE (Word Position Embedding)
is not used in the MeMs model.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.fn.dropout import dropout
from models_x.nn.embedding import Embedding
from models_x.mems.mems_config import MeMsConfig

__all__ = ["MeMsStem"]


@register_dataclass
@dataclass(frozen=True)
class MeMsStem():
    """
    Memory Mosaics Stem (WTE + WPE and dropout).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: MeMsConfig = field(metadata=metadata)
    wte: Embedding = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: MeMsConfig,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Config attributes
        vocab_size = int(kwargs.get('vocab_size', cfg.vocab_size))
        d_model = int(kwargs.get('d_model', cfg.d_model))
        init_std = float(kwargs.get('init_std', cfg.init_std))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Word Token Embedding (WTE)
        wte = Embedding(num_embeddings=vocab_size,
                        embedding_dim=d_model,
                        init_std=init_std,
                        dtype=dtype)

        return cls(cfg=cfg, wte=wte)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # Params dict
        return {'wte': self.wte.init_params(key=key)}

    def __call__(self: Self,
                 params: dict[str, Any],
                 input_ids: Int[Array, "B T"],              # noqa: F722
                 key: Array | None = None,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # Embeddings
        tok_emb = self.wte(params=params['wte'],
                           arr=input_ids)                   # B x T x D

        # Dropout
        if not deterministic:
            p = float(self.cfg.p_drop_stem)
            tok_emb = dropout(arr=tok_emb, key=key, p=p)    # B x T x D

        return tok_emb                                      # B x T x D
