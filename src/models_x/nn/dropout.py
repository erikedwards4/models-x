"""
JAX functional equivalent of torch.nn.Dropout.
"""

import jax
from jaxtyping import Float, Array

__all__ = ["dropout"]


def dropout(batch: Float[Array, "..."],
            *,
            p: float = 0.0,
            key: Array | None = None,
            deterministic: bool = True,
            ) -> Float[Array, "..."]:
    """
    Applies usual (element-wise) Dropout if training.
    Masks and scales to maintain the same expected value.
    """
    # Only if training and valid
    if not deterministic and isinstance(key, Array) and 0.0 < p < 0.999999:
        # Keep probability
        p_keep = 1.0 - p

        # Bernoulli mask
        mask = jax.random.bernoulli(key=key,
                                    p=p_keep,
                                    shape=batch.shape)

        # Apply mask and scale
        batch = (batch * mask) / p_keep

    return batch
