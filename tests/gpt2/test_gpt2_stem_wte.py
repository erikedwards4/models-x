"""
Pytest function for gpt2/gpt2_stem_wte.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.profile_callable import profile_callable
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_stem_wte import GPT2StemWTE


# gpt2_stem_wte.GPT2StemWTE
@pytest.mark.parametrize("vocab_size", (50257, ))
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_stem_wte(vocab_size, d_model, dtype):
    """
    Pytest gpt2_stem_wte.GPT2StemWTE.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    config = GPT2Config(vocab_size=vocab_size,
                        d_model=d_model,
                        dtype=dtype)

    # Get mdl
    wte = GPT2StemWTE(cfg=config)
    assert isinstance(wte, GPT2StemWTE)
    assert callable(wte)
    assert wte.cfg.dtype == config.dtype == dtype

    # Get PRNG key
    prng_key = jax.random.PRNGKey(seed=0)

    # Get params dict
    params = wte.init_params(prng_key=prng_key)
    assert 'wte' in params
    assert isinstance(params['wte'], Array)
    assert params['wte'].dtype == jnp.dtype(config.dtype)
    assert 0.0 < jnp.std(params['wte']).item() < 1.0

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = (nbatch, ntoks)
    input_ids = jax.random.randint(key=prng_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)
    print(f"input_ids.device = {input_ids.device}")

    # Test __call__
    batch_out = wte(input_ids=input_ids,
                    params=params)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == jnp.dtype(config.dtype)
    assert batch_out.device == input_ids.device
    assert batch_out.shape == (nbatch, ntoks, config.d_model)
    assert jnp.all(jnp.isfinite(batch_out))

    # Test embedding look-up
    embd0 = params['wte'][input_ids[0, 0]]
    assert jnp.allclose(batch_out[0, 0], embd0)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=wte,
                     batch_in=input_ids,
                     params=params)
