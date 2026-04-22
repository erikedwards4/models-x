"""
Pytest function for nn/linear.py.
"""
from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.nn.linear import Linear


# linear.Linear
@pytest.mark.parametrize("in_features", (768, ))
@pytest.mark.parametrize("out_features", (32, ))
@pytest.mark.parametrize("bias", (True, ))
@pytest.mark.parametrize("w_init", ('uniform', ))
@pytest.mark.parametrize("b_init", ('zeros', ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_linear(in_features, out_features, bias,
                w_init, b_init, dtype):
    """
    Pytest linear.Linear.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get instance
    mdl = Linear(in_features=in_features,
                 out_features=out_features,
                 bias=bias,
                 w_init=w_init,
                 b_init=b_init,
                 dtype=dtype)
    assert isinstance(mdl, Linear)
    assert callable(mdl)
    assert mdl.dtype == dtype

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, data_key = jax.random.split(key=prng_key, num=2)

    # Get params dict
    params = mdl.init_params(key=params_key)
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
    ntoks = 16
    size_in = (nbatch, ntoks, in_features)
    batch_in = jax.random.normal(key=data_key,
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
    assert batch_out.shape == (nbatch, ntoks, out_features)
    assert jnp.all(jnp.isfinite(batch_out))

    # JIT compile
    mdl_jit = jax.jit(mdl,
                      static_argnames=())
    batch_out = mdl_jit(params=params,
                        arr=batch_in)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in)

    # JAXPR
    fun = partial(mdl,
                  params=params,
                  arr=batch_in)
    jaxpr = jax.make_jaxpr(fun=fun)()
    print(f"JAXPR:\n{jaxpr}")
