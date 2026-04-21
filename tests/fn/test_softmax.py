"""
Pytest function for nn/softmax.py.
"""
from functools import partial
import pytest
import jax
import jax.numpy as jnp
# from jax.nn import softmax
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.fn.softmax import softmax


# softmax.softmax
@pytest.mark.parametrize("axis", (-1, ))
def test_softmax(axis):
    """
    Pytest softmax.softmax.
    """
    # Start
    print("")
    print_memory_stats(label="start")
    prng_key = jax.random.PRNGKey(seed=0)

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 16
    d_model = 768
    size_in = (nbatch, ntoks, d_model)
    dtype = jnp.float32
    device = jax.devices("gpu")[0]
    batch_in = jax.random.normal(key=prng_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = softmax(arr=batch_in, axis=axis)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == dtype
    assert batch_out.device == batch_in.device
    assert batch_out.shape == batch_in.shape
    assert jnp.all(jnp.isfinite(batch_out))

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=softmax,
                     n_runs=64,
                     arr=batch_in,
                     axis=axis)

    # JAXPR
    fun = partial(softmax, arr=batch_in, axis=axis)
    jaxpr = jax.make_jaxpr(fun=fun)()
    print(f"JAXPR:\n{jaxpr}")
