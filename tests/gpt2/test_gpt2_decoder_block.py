"""
Pytest function for gpt2/gpt2_decoder_block.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_decoder_block import GPT2DecoderBlock


# gpt2_decoder_block.GPT2DecoderBlock
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("attn_implementation", ("sdpa", ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_decoder_block(d_model, attn_implementation, dtype):
    """
    Pytest gpt2_decoder_block.GPT2DecoderBlock.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = GPT2Config(d_model=d_model,
                     attn_implementation=attn_implementation,
                     dtype=dtype)

    # Get mdl
    blk = GPT2DecoderBlock.from_config(cfg=cfg)
    assert isinstance(blk, GPT2DecoderBlock)
    assert callable(blk)
    assert blk.cfg.dtype == cfg.dtype == dtype

    # Get PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, dropout_key = jax.random.split(key=prng_key, num=2)

    # Get params dict
    params = blk.init_params(key=params_key)
    assert 'wte' in params
    assert isinstance(params['wte'], dict)

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
    batch_out = blk(params=params,
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
    blk_jit = jax.jit(blk,
                      static_argnames=("deterministic",))
    batch_out = blk_jit(params=params,
                        arr=batch_in,
                        key=dropout_key,
                        deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=blk_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in,
                     key=None,
                     deterministic=True)
