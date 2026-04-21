"""
Pytest function for gpt2/gpt2_decoder_block_mlp.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_decoder_block_mlp import GPT2DecoderBlockMLP


# gpt2_decoder_block_mlp.GPT2DecoderBlockMLP
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_decoder_block_mlp(d_model, dtype):
    """
    Pytest gpt2_decoder_block_mlp.GPT2DecoderBlockMLP.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = GPT2Config(d_model=d_model,
                     dtype=dtype)

    # Get mdl
    mlp = GPT2DecoderBlockMLP.from_config(cfg=cfg)
    assert isinstance(mlp, GPT2DecoderBlockMLP)
    assert callable(mlp)
    assert mlp.cfg.dtype == cfg.dtype == dtype

    # Get PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, dropout_key = jax.random.split(key=prng_key, num=2)

    # Get params dict
    params = mlp.init_params(key=params_key)
    assert 'c_proj' in params
    assert isinstance(params['c_proj'], dict)

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 16
    size_in = (nbatch, ntoks, d_model)
    batch_in = jax.random.normal(key=prng_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = mlp(params=params,
                    arr=batch_in,
                    key=dropout_key,
                    deterministic=False)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == batch_in.dtype
    assert batch_out.device == batch_in.device
    assert batch_out.shape == batch_in.shape
    assert jnp.all(jnp.isfinite(batch_out))

    # JIT compile
    mlp_jit = jax.jit(mlp,
                      static_argnames=("deterministic",))
    batch_out = mlp_jit(params=params,
                        arr=batch_in,
                        key=dropout_key,
                        deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mlp_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in,
                     key=dropout_key,
                     deterministic=True)
