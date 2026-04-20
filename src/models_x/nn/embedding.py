"""
GPT-2 WTE (Word Token Embedding) part of the stem.

This is a look-up from integer token IDs in [0, vocab_size)
to float embedding vectors with length d_model.
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, DTypeLike

__all__ = ["Embedding"]


@register_dataclass
@dataclass
class Embedding():
    """
    JAX dataclass equivalent of torch.nn.Embedding,
    but simplified (no max_norm, sparse, etc.).
    """
    # Metadata (for jax.tree_util setup)
    metadata = dict(static=True)    # pylint: disable=use-dict-literal

    # Self attributes
    num_embeddings: int = field(default=1, metadata=metadata)
    embedding_dim: int = field(default=1, metadata=metadata)
    init_std: float = field(default=0.02, metadata=metadata)
    dtype: DTypeLike = field(default=jnp.float32, metadata=metadata)

    def init_params(self: Self,
                    prng_key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Embedding weights shape and std
        shape = (int(self.num_embeddings),
                 int(self.embedding_dim))
        std = float(self.init_std)

        # Embedding weights
        weight = jax.random.normal(key=prng_key,
                                   shape=shape,
                                   dtype=self.dtype,
                                   ) * std          # N x D

        return {'w': weight}

    def __call__(self: Self,
                 input_ids: Int[Array, "..."],      # noqa: F722
                 params: dict[str, Array],
                 ) -> Float[Array, "... D"]:        # noqa: F722
        """
        D = embedding_dim (len of each embedding vec)
        """
        # torch.nn.Embedding lookup --> jnp.take
        return jnp.take(a=params['w'],
                        indices=input_ids,
                        axis=0,
                        mode='fill',
                        fill_value=0.0,
                        unique_indices=False,
                        indices_are_sorted=False)   # ... x D
