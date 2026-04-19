"""
Pytest function for gpt2/gpt2_stem_wpe.py.
"""
import time
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.utils.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_stem_wpe import GPT2StemWPE


# gpt2_stem_wpe.GPT2StemWPE
@pytest.mark.parametrize("n_positions", (1024, ))
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_gpt2_stem_wpe(n_positions, d_model, dtype):
    """
    Pytest gpt2_stem_wpe.GPT2StemWPE.
    """
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get config
    config = GPT2Config(n_positions=n_positions,
                        d_model=d_model,
                        dtype=dtype)

    # Get mdl
    wpe = GPT2StemWPE(cfg=config)
    assert isinstance(wpe, GPT2StemWPE)
    assert callable(wpe)
    assert wpe.cfg.dtype == config.dtype == dtype

    # Get PRNG key
    prng_key = jax.random.PRNGKey(seed=0)

    # Get params dict
    params = wpe.init_params(prng_key=prng_key)
    assert 'wpe' in params
    assert isinstance(params['wpe'], Array)
    assert params['wpe'].dtype == jnp.dtype(config.dtype)
    assert 0.0 < jnp.std(params['wpe']).item() < 1.0

    # Check device (should default to GPU if using jaxlib)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"params.devices() = {sample_leaf.devices()}")

    # Make input data
    ntoks = 16
    position_ids = jnp.arange(start=0,
                              stop=ntoks,
                              step=1,
                              dtype=jnp.int32,
                              device=device)
    print(f"position_ids.device = {position_ids.device}")

    # Test __call__
    batch_out = wpe(position_ids=position_ids,
                    params=params)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == jnp.dtype(config.dtype)
    assert batch_out.device == position_ids.device
    assert batch_out.shape == (ntoks, config.d_model)
    assert jnp.all(jnp.isfinite(batch_out))

    # Test embedding look-up
    embd0 = params['wpe'][position_ids[0]]
    assert jnp.allclose(batch_out[0], embd0)

    # Profile

    # Warmup (first jit call triggers compilation)
    for _ in range(4):
        batch_out = wpe(position_ids=position_ids,
                        params=params)
        batch_out.block_until_ready()

    # Actual profiled runs
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        batch_out = wpe(position_ids=position_ids,
                        params=params)
        batch_out.block_until_ready()
        times.append((time.perf_counter() - t0)*1000)
    times = jnp.array(times)
    print(f"et mean: {jnp.mean(times):.6f} ms")
    print(f"et med : {jnp.median(times):.6f} ms")
    print(f"et min : {jnp.min(times):.6f} ms")

    # See memory usage
    print_memory_stats(label="after")
