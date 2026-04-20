"""
Pytest function for nn/layer_norm.py.
"""
from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.nn.layer_norm import LayerNorm


# layer_norm.LayerNorm
@pytest.mark.parametrize("normalized_shape", (768, ))
@pytest.mark.parametrize("eps", (1e-5, ))
@pytest.mark.parametrize("bias", (True, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_layer_norm(normalized_shape, eps, bias, dtype):
    """
    Pytest layer_norm.LayerNorm.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get instance
    mdl = LayerNorm(normalized_shape=normalized_shape,
                    eps=eps,
                    bias=bias,
                    dtype=dtype)
    assert isinstance(mdl, LayerNorm)
    assert callable(mdl)
    assert mdl.dtype == dtype

    # Get PRNG key
    prng_key = jax.random.PRNGKey(seed=0)

    # Get params dict
    params = mdl.init_params(prng_key)
    assert 'w' in params
    assert isinstance(params['w'], Array)
    assert params['w'].dtype == dtype
    assert 0.0 <= jnp.std(params['w']).item() < 100.0
    if bias:
        assert 'b' in params
        assert isinstance(params['b'], Array)
        assert params['b'].dtype == dtype
        assert 0.0 <= jnp.std(params['b']).item() < 100.0

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = (nbatch, ntoks, normalized_shape)
    batch_in = jax.random.normal(key=prng_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)
    print(f"batch_in.device = {batch_in.device}")

    # Test __call__
    batch_out = mdl(params=params,
                    arr=batch_in)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == batch_in.dtype == dtype
    assert batch_out.device == batch_in.device
    assert batch_out.shape == batch_in.shape
    assert jnp.all(jnp.isfinite(batch_out))

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl,
                     n_runs=64,
                     params=params,
                     arr=batch_in)

    # JAXPR
    fun = partial(mdl,
                  params=params,
                  arr=batch_in)
    jaxpr = jax.make_jaxpr(fun=fun)()
    print(f"JAXPR:\n{jaxpr}")
