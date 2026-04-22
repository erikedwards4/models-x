"""
Pytest function for nn/embedding.py.
"""
from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.nn.embedding import Embedding


# embedding.Embedding
@pytest.mark.parametrize("num_embeddings", (50257, ))
@pytest.mark.parametrize("embedding_dim", (768, ))
@pytest.mark.parametrize("init_std", (0.02, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_embedding(num_embeddings, embedding_dim,
                   init_std, dtype):
    """
    Pytest embedding.Embedding.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get instance
    mdl = Embedding(num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    init_std=init_std,
                    dtype=dtype)
    assert isinstance(mdl, Embedding)
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
    assert 0.0 < jnp.std(params['w']).item() < 1.0

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = (nbatch, ntoks)
    input_ids = jax.random.randint(key=data_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=num_embeddings,
                                   dtype=jnp.int32,
                                   ).to_device(device)
    print(f"input_ids.device = {input_ids.device}")

    # Test __call__
    batch_out = mdl(params=params,
                    arr=input_ids)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == dtype
    assert batch_out.device == input_ids.device
    assert batch_out.shape == (nbatch, ntoks, embedding_dim)
    assert jnp.all(jnp.isfinite(batch_out))

    # Test embedding look-up
    embd0 = params['w'][input_ids[0, 0]]
    assert jnp.allclose(batch_out[0, 0], embd0)

    # JIT compile
    mdl_jit = jax.jit(mdl,
                      static_argnames=())
    batch_out = mdl_jit(params=params,
                        arr=input_ids)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl_jit,
                     n_runs=32,
                     params=params,
                     arr=input_ids)

    # JAXPR
    fun = partial(mdl,
                  params=params,
                  arr=input_ids)
    jaxpr = jax.make_jaxpr(fun=fun)()
    print(f"JAXPR:\n{jaxpr}")
