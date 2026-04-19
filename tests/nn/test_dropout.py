"""
Pytest function for nn/dropout.py.
"""
import time
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.nn.dropout import dropout


# dropout.dropout
@pytest.mark.parametrize("p", (0.0, ))
@pytest.mark.parametrize("deterministic", (False, ))
def test_dropout(p, deterministic):
    """
    Pytest dropout.dropout.
    """
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get PRNG key
    prng_key = jax.random.PRNGKey(seed=0)

    # Make input data
    nbatch = 4          # micro-batch size
    nsamps = 16
    size_in = (nbatch, nsamps)
    dtype = jnp.float32
    batch_in = jax.random.normal(key=prng_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = dropout(batch=batch_in,
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

    # Profile

    # Warmup (first jit call triggers compilation)
    for _ in range(4):
        batch_out = dropout(batch=batch_in,
                            p=p,
                            key=prng_key,
                            deterministic=deterministic)
        batch_out.block_until_ready()

    # Actual profiled runs
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        batch_out = dropout(batch=batch_in,
                            p=p,
                            key=prng_key,
                            deterministic=deterministic)
        batch_out.block_until_ready()
        times.append((time.perf_counter() - t0)*1000)
    times = jnp.array(times)
    print(f"et mean: {jnp.mean(times):.6f} ms")
    print(f"et med : {jnp.median(times):.6f} ms")
    print(f"et min : {jnp.min(times):.6f} ms")

    # See memory usage
    print_memory_stats(label="after")
