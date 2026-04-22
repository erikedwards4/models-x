"""
JAX class equivalent to torch.nn.LayerNorm,
but simplified to assume elementwise_affine=True
(without elementwise_affine a layer norm is just
a z-score along axis=-1, so rarely used as such).
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, DTypeLike

__all__ = ["LayerNorm"]


@register_dataclass
@dataclass(frozen=True)
class LayerNorm():
    """
    normalized_shape: shape along the axis=-1 to be normalized
    bias: if to include bias
    dtype: jnp data type
    """
    # Metadata (for jax.tree_util)
    metadata = dict(static=True)    # pylint: disable=use-dict-literal

    # Self attributes
    normalized_shape: int = field(metadata=metadata)
    eps: float = field(default=1e-5, metadata=metadata)
    bias: bool = field(default=True, metadata=metadata)
    dtype: DTypeLike = field(default=jnp.float32, metadata=metadata)

    def __post_init__(self: Self,
                      ) -> None:
        """
        Standard dataclass method, called automatically.
        """
        # Checks
        if self.normalized_shape < 1:
            raise ValueError("normalized_shape must be a positive int")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("eps out of expected range")

    def init_params(self: Self,
                    _key: Array | None = None,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Shape
        shape = (int(self.normalized_shape), )

        # LayerNorm weight (gamma)
        gamma = jnp.ones(shape=shape,
                         dtype=self.dtype)         # D
        params = {'w': gamma}

        # LayerNorm bias (beta)
        if self.bias:
            beta = jnp.zeros(shape=shape,
                             dtype=self.dtype)     # D
            params['b'] = beta

        return params

    def __call__(self: Self,
                 params: dict[str, Array],
                 arr: Float[Array, "... D"],        # noqa: F722
                 ) -> Float[Array, "... D"]:        # noqa: F722
        """
        D = normalized_shape (dim along axis=-1)
        """
        # Zero mean over last dim
        mean = jnp.mean(a=arr,
                        axis=-1,
                        keepdims=True)              # ... x 1
        arr0 = arr - mean                           # ... x D

        # Unit std over last dim
        var = jnp.mean(a=arr0**2,
                       axis=-1,
                       keepdims=True)               # ... x 1
        istd = jax.lax.rsqrt(var + self.eps)        # ... x 1
        arr1 = arr0 * istd                          # ... x D

        # Scale by weight (gamma)
        arr_out = arr1 * params['w']                # ... x D

        # Shift by bias (beta)
        if self.bias:
            arr_out = arr_out + params['b']         # ... x D

        return arr_out                              # ... x D
