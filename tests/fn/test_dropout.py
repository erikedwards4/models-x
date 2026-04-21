"""
Pytest function for nn/dropout.py.
"""
# from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.fn.dropout import dropout


# dropout.dropout
@pytest.mark.parametrize("p", (0.1, ))
@pytest.mark.parametrize("deterministic", (False, ))
def test_dropout(p, deterministic):
    """
    Pytest dropout.dropout.
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
    batch_out = dropout(arr=batch_in,
                        p=p,
                        key=prng_key,
                        deterministic=deterministic)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == dtype
    assert batch_out.device == batch_in.device
    assert batch_out.shape == batch_in.shape
    assert jnp.all(jnp.isfinite(batch_out))
    if deterministic or not 0.0 < p < 0.999999:
        assert jnp.allclose(a=batch_in,
                            b=batch_out,
                            rtol=1e-7,
                            atol=1e-7)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=dropout,
                     n_runs=32,
                     arr=batch_in,
                     p=p,
                     key=prng_key,
                     deterministic=deterministic)

    # JAXPR
    # fun = partial(dropout,
    #               arr=batch_in,
    #               p=p,
    #               key=prng_key,
    #               deterministic=deterministic)
    # jaxpr = jax.make_jaxpr(fun=fun)()
    # print(f"JAXPR:\n{jaxpr}")
