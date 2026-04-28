"""
Pytest training test for gpt2/gpt2.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2 import GPT2


# gpt2.GPT2
@pytest.mark.parametrize("nblocks", (6, ))
@pytest.mark.parametrize("attn_implementation", ("eager", ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def train_gpt2(nblocks, attn_implementation, dtype):
    """
    Pytest train gpt2.GPT2.
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
    mdl = GPT2.from_config(cfg=cfg)
    assert isinstance(mdl, GPT2)
    assert callable(mdl)
    assert mdl.cfg.dtype == cfg.dtype == dtype

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, data_key, call_key = \
        jax.random.split(key=prng_key, num=3)

    # Get params dict
    params = mdl.init_params(key=params_key)
    assert 'stem' in params
    assert isinstance(params['stem'], dict)

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 512
    size_in = (nbatch, ntoks)
    input_ids = jax.random.randint(key=data_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=cfg.vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)

    # Test __call__
    batch_out = mdl(params=params,
                    input_ids=input_ids,
                    key=call_key,
                    deterministic=False)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == jnp.dtype(cfg.dtype)
    assert batch_out.device == input_ids.device
    assert batch_out.shape == (nbatch, ntoks, cfg.d_model)
    assert jnp.all(jnp.isfinite(batch_out))

    # JIT compile
    mdl_jit = jax.jit(mdl,
                      static_argnames=("deterministic",))
    batch_out = mdl_jit(params=params,
                        input_ids=input_ids,
                        key=call_key,
                        deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl_jit,
                     n_runs=32,
                     params=params,
                     input_ids=input_ids,
                     key=call_key,
                     deterministic=True)
