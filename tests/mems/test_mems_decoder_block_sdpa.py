"""
Pytest function for mems/mems_decoder_block_sdpa.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.mems.mems_config import MeMsConfig
from models_x.mems.mems_decoder_block_sdpa import MeMsDecoderBlockSDPA


# mems_decoder_block_sdpa.MeMsDecoderBlockSDPA
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_mems_decoder_block_sdpa(d_model, dtype):
    """
    Pytest mems_decoder_block_sdpa.MeMsDecoderBlockSDPA.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = MeMsConfig(d_model=d_model,
                     dtype=dtype)

    # Get mdl
    mha = MeMsDecoderBlockSDPA.from_config(cfg=cfg)
    assert isinstance(mha, MeMsDecoderBlockSDPA)
    assert callable(mha)
    assert mha.cfg.dtype == cfg.dtype == dtype

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, data_key, call_key = \
        jax.random.split(key=prng_key, num=3)

    # Get params dict
    params = mha.init_params(key=params_key)
    assert 'out_proj' in params
    assert isinstance(params['out_proj'], dict)

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 512
    size_in = (nbatch, ntoks, d_model)
    batch_in = jax.random.normal(key=data_key,
                                 shape=size_in,
                                 dtype=dtype,
                                 ).to_device(device)

    # Test __call__
    batch_out = mha(params=params,
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
    mha_jit = jax.jit(mha,
                      static_argnames=("deterministic",))
    batch_out = mha_jit(params=params,
                        arr=batch_in,
                        key=call_key,
                        deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mha_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in,
                     key=call_key,
                     deterministic=True)
