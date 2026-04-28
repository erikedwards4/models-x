"""
Pytest function for gpt2/gpt2_lm.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float
from models_x.util.profile_callable import profile_callable
from models_x.util.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_lm import GPT2LM


# gpt2_lm.GPT2LM
@pytest.mark.parametrize("nblocks", (2, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_lm(nblocks, dtype):
    """
    Pytest gpt2_lm.GPT2LM.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = GPT2Config(nblocks=nblocks,
                     dtype=dtype)

    # Get mdl
    mdl = GPT2LM.from_config(cfg=cfg)
    assert isinstance(mdl, GPT2LM)
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
    ntoks = 1024
    size_in = (nbatch, ntoks)
    token_ids = jax.random.randint(key=data_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=cfg.vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)

    # Test __call__
    logits = mdl(params=params,
                 token_ids=token_ids,
                 key=call_key,
                 deterministic=False)
    print(f"logits.dtype = {logits.dtype}")
    print(f"logits.shape = {logits.shape}")
    assert isinstance(logits, Float[jnp.ndarray, "..."])
    assert logits.dtype == jnp.dtype(cfg.dtype)
    assert logits.device == token_ids.device
    assert logits.shape == (nbatch, ntoks, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))

    # JIT compile
    mdl_jit = jax.jit(mdl,
                      static_argnames=("deterministic",))
    logits = mdl_jit(params=params,
                     token_ids=token_ids,
                     key=call_key,
                     deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl_jit,
                     n_runs=32,
                     params=params,
                     token_ids=token_ids,
                     key=call_key,
                     deterministic=True)
