"""
JAX class for ~equivalent to torch.nn.LayerNorm,
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, DTypeLike

__all__ = ["LayerNorm"]


@register_dataclass
@dataclass
class LayerNorm():
    """
    JAX dataclass equivalent of torch.nn.LayerNorm.
    """
    # Metadata (for jax.tree_util setup)
    metadata = dict(static=True)    # pylint: disable=use-dict-literal

    # Self attributes
    in_features: int = field(default=1, metadata=metadata)
    out_features: int = field(default=1, metadata=metadata)
    bias: bool = field(default=True, metadata=metadata)
    w_init: str = field(default="uniform", metadata=metadata)
    b_init: str = field(default="zeros", metadata=metadata)
    dtype: DTypeLike = field(default=jnp.float32, metadata=metadata)

    def init_params(self: Self,
                    prng_key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        k1, k2 = jax.random.split(prng_key)

        # LayerNorm weight
        w_shape = (int(self.in_features),
                   int(self.out_features))
        maxval = 1 / jnp.sqrt(self.in_features)
        weight = jax.random.uniform(key=k1,
                                    shape=w_shape,
                                    minval=-maxval,
                                    maxval=maxval,
                                    dtype=self.dtype) \
            if self.w_init == "uniform" else \
            jnp.zeros(shape=w_shape,
                      dtype=self.dtype)                 # Ci x Co
        params = {'w': weight}

        # LayerNorm bias
        if self.bias:
            b_shape = (int(self.out_features), )
            maxval = 1 / jnp.sqrt(self.out_features)
            bias = jax.random.uniform(key=k2,
                                      shape=b_shape,
                                      minval=-maxval,
                                      maxval=maxval,
                                      dtype=self.dtype) \
                if self.b_init == "uniform" else \
                jnp.zeros(shape=b_shape,
                          dtype=self.dtype)             # Co
            params['b'] = bias

        return params

    def __call__(self: Self,
                 batch: Float[Array, "... Ci"],         # noqa: F722
                 params: dict[str, Array],
                 ) -> Float[Array, "... Co"]:           # noqa: F722
        """
        Ci = in_features (nchans_in)
        Co = out_features (nchans_out)
        """
        # JAX handles the dot product across
        # leading '...' dims automatically
        batch_out = jnp.dot(batch, params['w'])         # ... x Co

        if self.bias:
            batch_out = batch_out + params['b']         # ... x Co

        return batch_out
