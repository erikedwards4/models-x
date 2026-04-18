"""
GPT-2 WTE (Word Token Embedding) part of the stem.

This is a look-up from integer token IDs in [0, vocab_size)
to float embedding vectors with length d_model.
"""

from typing import Self
from dataclasses import dataclass
import jax.numpy as jnp
from jax._src.numpy.scalar_types import _ScalarMeta

__all__ = ["GPT2Config"]


@dataclass(frozen=True)
class GPT2Config:
    """
    Configs for GPT-2 model.
    """
    # General
    dtype: _ScalarMeta = jnp.bfloat16

    # Architecture dimensions
    vocab_size: int = 50257
    n_positions: int = 1024
    d_model: int = 768
    nblocks: int = 12
    nheads: int = 12

    # Regularization & precision
    p_drop_stem: float = 0.1
    p_drop_attn: float = 0.1
    p_drop_res: float = 0.1
    p_drop_mlp: float = 0.1
    lnorm_eps: float = 1e-5

    # Helper for head dimension
    @property
    def d_head(self: Self,
               ) -> int:
        """Sets d_head."""
        assert self.d_model % self.nheads == 0, \
            "d_model must be divisible by nheads"
        return self.d_model // self.nheads
