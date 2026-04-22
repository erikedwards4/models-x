"""
Pytest function for gpt2/gpt2_stem.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_stem import GPT2Stem


# gpt2_stem.GPT2Stem
@pytest.mark.parametrize("vocab_size", (50257, ))
@pytest.mark.parametrize("n_positions", (512, ))
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_stem(vocab_size, n_positions, d_model, dtype):
    """
    Pytest gpt2_stem.GPT2Stem.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    cfg = GPT2Config(vocab_size=vocab_size,
                     n_positions=n_positions,
                     d_model=d_model,
                     dtype=dtype)

    # Get mdl
    stem = GPT2Stem.from_config(cfg=cfg)
    assert isinstance(stem, GPT2Stem)
    assert callable(stem)
    assert stem.cfg.dtype == cfg.dtype == dtype

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, data_key, call_key = \
        jax.random.split(key=prng_key, num=3)

    # Get params dict
    params = stem.init_params(key=params_key)
    assert 'wte' in params
    assert isinstance(params['wte'], dict)

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
                                   maxval=vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)

    # Test __call__
    batch_out = stem(params=params,
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
    stem_jit = jax.jit(stem,
                       static_argnames=("deterministic",))
    batch_out = stem_jit(params=params,
                         input_ids=input_ids,
                         key=call_key,
                         deterministic=False)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=stem_jit,
                     n_runs=32,
                     params=params,
                     input_ids=input_ids,
                     key=call_key,
                     deterministic=True)
