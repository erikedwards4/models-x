"""
Utility function to profile a Callable.

This is meant for a function (fun) that takes
a JAX Array as input (arr), along with an
optional list of kwargs, and gives a similar
JAX Array as output (arr_out).

The fun can be an instance of a class that has
a __call__ method (and is therefore a Callable).

Usually only used during testing and profiling.
"""

from typing import Any, Callable
import time
import jax.numpy as jnp

__all__ = ["profile_callable"]


def profile_callable(fun: Callable,
                     *,
                     n_runs: int,
                     **kwargs: Any,
                     ) -> None:
    """
    Util function to profile a Callable,
    using arr and kwargs as inputs.

    The fun must support:
    arr_out = fun(**kwargs);
    (so all positional args must allow kwargs entry)

    Does n_runs//4 warm-up runs, and then
    the rest of n_runs to get the stats.
    """
    # Check
    assert callable(fun)

    # Profile
    times: list[float] = []
    # Warmup (first jit call triggers compilation)
    for _ in range(n_runs//4):
        arr_out = fun(**kwargs)
        arr_out.block_until_ready()

    # Actual profiled runs
    for _ in range(n_runs-n_runs//4):
        t0 = time.perf_counter()
        arr_out = fun(**kwargs)
        arr_out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    # Print
    times_array = jnp.array(times)
    print(f"et mean: {jnp.mean(times_array):.6f} ms")
    print(f"et med : {jnp.median(times_array):.6f} ms")
    print(f"et min : {jnp.min(times_array):.6f} ms")
