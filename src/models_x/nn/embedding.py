"""
JAX class equivalent to torch.nn.Embedding,
but simplified (no max_norm, sparse, etc.).
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, DTypeLike

__all__ = ["Embedding"]


@register_dataclass
@dataclass(frozen=True)
class Embedding():
    """
    num_embeddings: num embedding vecs
    embedding_dim: len of embedding vecs
    init_std: initialize standard dev
    dtype: jnp data type
    """
    # Metadata (for jax.tree_util)
    metadata = dict(static=True)    # pylint: disable=use-dict-literal

    # Self attributes
    num_embeddings: int = field(metadata=metadata)
    embedding_dim: int = field(metadata=metadata)
    init_std: float = field(default=0.02, metadata=metadata)
    dtype: DTypeLike = field(default=jnp.float32, metadata=metadata)

    def __post_init__(self: Self,
                      ) -> None:
        """
        Standard dataclass method, called automatically.
        """
        # Checks
        if self.num_embeddings < 1:
            raise ValueError("num_embeddings must be a positive int")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive int")
        if not 1e-6 < self.init_std < 1e2:
            raise ValueError("init_std out of expected range")

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Embedding weights shape and std
        shape = (int(self.num_embeddings),
                 int(self.embedding_dim))
        std = float(self.init_std)

        # Embedding weights
        weight = jax.random.normal(key=key,
                                   shape=shape,
                                   dtype=self.dtype,
                                   ) * std              # N x D

        return {'w': weight}

    def __call__(self: Self,
                 params: dict[str, Array],
                 arr: Int[Array, "..."],                # noqa: F722
                 ) -> Float[Array, "... D"]:            # noqa: F722
        """
        D = embedding_dim (len of each embedding vec)
        """
        # torch.nn.Embedding lookup --> jnp.take
        return jnp.take(a=params['w'],
                        indices=arr,
                        axis=0,
                        mode=None,
                        unique_indices=False,
                        indices_are_sorted=False)       # ... x D
