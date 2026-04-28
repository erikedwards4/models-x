"""
Pytest function for gpt2/gpt2_decoder.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.util.profile_callable import profile_callable
from models_x.util.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_decoder import GPT2Decoder


# gpt2_decoder.GPT2Decoder
@pytest.mark.parametrize("nblocks", (2, ))
@pytest.mark.parametrize("attn_implementation", ("eager", ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_decoder(nblocks, attn_implementation, dtype):
    """
    Pytest gpt2_decoder.GPT2Decoder.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = GPT2Config(nblocks=nblocks,
                     attn_implementation=attn_implementation,
                     dtype=dtype)

    # Get mdl
    decoder = GPT2Decoder.from_config(cfg=cfg)
    assert isinstance(decoder, GPT2Decoder)
    assert callable(decoder)
    assert decoder.cfg.dtype == cfg.dtype == dtype

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, data_key, call_key = \
        jax.random.split(key=prng_key, num=3)

    # Get params dict
    params = decoder.init_params(key=params_key)
    assert 'lnorm_f' in params
    assert isinstance(params['lnorm_f'], dict)

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = (nbatch, ntoks, cfg.d_model)
    batch_in = jax.random.normal(key=data_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = decoder(params=params,
                        arr=batch_in,
                        key=call_key,
                        deterministic=False)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == batch_in.dtype
    assert batch_out.device == batch_in.device
    assert batch_out.shape == batch_in.shape
    assert jnp.all(jnp.isfinite(batch_out))

    # JIT compile
    decoder_jit = jax.jit(decoder,
                          static_argnames=("deterministic",))
    batch_out = decoder_jit(params=params,
                            arr=batch_in,
                            key=call_key,
                            deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=decoder_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in,
                     key=call_key,
                     deterministic=True)
