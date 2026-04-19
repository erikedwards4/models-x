"""
Utility function to profile a Callable.

This is meant for a function (fun) that takes
a JAX Array as input (batch_in), along with an
optional list of kwargs, and gives a similar
JAX Array as output (batch_out).

The fun can be an instance of a class that has
a __call__ method (and is therefore a Callable).

Usually only used during testing and profiling.
"""

from typing import Any, Callable
import time
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["profile_callable"]


def profile_callable(fun: Callable,
                     *,
                     batch_in: Array,
                     **kwargs: Any,
                     ) -> None:
    """
    Util function to profile a Callable,
    using batch_in and kwargs as inputs.

    The fun must support:
    batch_out = fun(batch_in, **kwargs);
    """
    # Check
    assert callable(fun)

    # Warmup (first jit call triggers compilation)
    for _ in range(4):
        batch_out = fun(batch_in, **kwargs)
        batch_out.block_until_ready()

    # Actual profiled runs
    times: list[float] = []
    for _ in range(8):
        t0 = time.perf_counter()
        batch_out = fun(batch_in, **kwargs)
        batch_out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    # Print
    times_array = jnp.array(times)
    print(f"et mean: {jnp.mean(times_array):.6f} ms")
    print(f"et med : {jnp.median(times_array):.6f} ms")
    print(f"et min : {jnp.min(times_array):.6f} ms")
