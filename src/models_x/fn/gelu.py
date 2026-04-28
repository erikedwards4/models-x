"""
JAX functional equivalent of torch.nn.GELU with approximate=False,
which is called "gelu" in HuggingFace.

This is the original GELU using the Gaussian error function (erf),
not the "new_gelu" using the tanh approximation (see gelu_new.py).

Note: the JAX compiler (XLA) does constant folding,
so isqrt2 = -np.sqrt(0.5) gets computed only once,
Thus, there is no performance gain to making a class,
and storing a self.isqrt2 = ... Also, it is idiomatic
JAX to use functions when no params dict.

However, just as fast (and matching how other compilers
work) is to define the constant _ISQRT2 at the module level.
"""

from jax.scipy.special import erf
from jaxtyping import Array, Float

__all__ = ["gelu"]

_ISQRT2 = -0.7071067811865476  # -1/sqrt(2)


def gelu(arr: Float[Array, "..."],
         ) -> Float[Array, "..."]:
    """
    arr: JAX Float Array of any shape
    """
    return 0.5 * arr * (1.0 + erf(_ISQRT2 * arr))   # ...
