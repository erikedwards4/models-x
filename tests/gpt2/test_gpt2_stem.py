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
@pytest.mark.parametrize("n_positions", (1024, ))
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
    config = GPT2Config(vocab_size=vocab_size,
                        n_positions=n_positions,
                        d_model=d_model,
                        dtype=dtype)

    # Get mdl
    stem = GPT2Stem(cfg=config)
    assert isinstance(stem, GPT2Stem)
    assert callable(stem)
    assert stem.cfg.dtype == config.dtype == dtype

    # Get PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    params_key, dropout_key = jax.random.split(prng_key)

    # Get params dict
    params = stem.init_params(prng_key=params_key)
    assert 'stem' in params
    assert isinstance(params['stem'], Array)
    assert params['stem'].dtype == jnp.dtype(config.dtype)
    assert 0.0 < jnp.std(params['stem']).item() < 1.0

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 16
    size_in = (nbatch, ntoks)
    input_ids = jax.random.randint(key=prng_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)

    # Test __call__
    batch_out = stem(input_ids=input_ids,
                     params=params,
                     dropout_key=dropout_key,
                     deterministic=False)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == jnp.dtype(config.dtype)
    assert batch_out.device == position_ids.device
    assert batch_out.shape == (nbatch, ntoks, config.d_model)
    assert jnp.all(jnp.isfinite(batch_out))

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=stem,
                     batch_in=input_ids,
                     params=params,
                     dropout_key=None,
                     deterministic=True)
