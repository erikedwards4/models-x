"""
Config for the GPT-2 model using JAX.

This uses dataclasses from the Python Standard Library,
as managed by jax.tree_util, which is the idiomatic JAX
approach. The configs herein propagate through the model
hierarchy as the central, single source of truth (with
overrides by kwargs possible for every class).

In the future, this could read from a config.yaml.
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
from jax.typing import DTypeLike
import jax.numpy as jnp

__all__ = ["GPT2Config"]


@register_dataclass
@dataclass(frozen=True)
class GPT2Config():
    """
    Configs for the GPT-2 model in JAX.
    """
    # Metadata (for jax.tree_util)
    metadata = dict(static=True)    # pylint: disable=use-dict-literal

    # General
    dtype: DTypeLike = field(default=jnp.float32, metadata=metadata)
    init_std: float = field(default=0.02, metadata=metadata)

    # Architecture dimensions
    vocab_size: int = field(default=50257, metadata=metadata)
    n_positions: int = field(default=1024, metadata=metadata)
    d_model: int = field(default=768, metadata=metadata)
    nblocks: int = field(default=12, metadata=metadata)
    nheads: int = field(default=12, metadata=metadata)

    # Regularization & precision
    p_drop_stem: float = field(default=0.1, metadata=metadata)
    p_drop_attn: float = field(default=0.1, metadata=metadata)
    p_drop_res: float = field(default=0.1, metadata=metadata)
    lnorm_eps: float = field(default=1e-5, metadata=metadata)

    # Attn implementation ('sdpa' vs. 'eager' or 'attn')
    # Note: 'sdpa' does not support attn_dropout for JAX
    attn_implementation: str = field(default="attn", metadata=metadata)

    # Helpers for derived properties
    @property
    def d_inner(self: Self) -> int:
        """Sets d_inner for the GPT2DecoderBlockMLP."""
        return 4 * self.d_model

    @property
    def d_head(self: Self) -> int:
        """Sets d_head for the GPT2DecoderBlockAtten."""
        assert self.d_model % self.nheads == 0, \
            "d_model must be divisible by nheads"
        return self.d_model // self.nheads

    @property
    def scale(self: Self) -> float:
        """Sets scale for the GPT2DecoderBlockAtten."""
        return self.d_model**-0.5
