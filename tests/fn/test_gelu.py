"""
Pytest function for nn/gelu.py.
"""
from functools import partial
import jax
import jax.numpy as jnp
# from jax.nn import gelu
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.fn.gelu import gelu


# gelu.gelu
def test_gelu():
    """
    Pytest gelu.gelu.
    """
    # Start
    print("")
    print_memory_stats(label="start")
    prng_key = jax.random.PRNGKey(seed=0)

    # Make input data
    nbatch = 8          # micro-batch size
    nsamps = 512
    size_in = (nbatch, nsamps)
    dtype = jnp.float32
    device = jax.devices("gpu")[0]
    batch_in = jax.random.normal(key=prng_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = gelu(batch_in)
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
    profile_callable(fun=gelu,
                     n_runs=64,
                     arr=batch_in)

    # JAXPR
    fun = partial(gelu, arr=batch_in)
    jaxpr = jax.make_jaxpr(fun=fun)()
    print(f"JAXPR:\n{jaxpr}")
