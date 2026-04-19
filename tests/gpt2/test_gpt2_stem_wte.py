"""
Pytest function for gpt2/gpt2_stem_wte.py.
"""
import time
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
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
    ntoks = 16
    size_in = (nbatch, ntoks)
    # device = "cuda"
    input_ids = jax.random.randint(key=prng_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=vocab_size,
                                   dtype=jnp.int32,
                                   ).to_device(device)
    print(f"input_ids.device = {input_ids.device}")

    # Test __call__
    batch_out = wte(params=params,
                    input_ids=input_ids)
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

    # Profile

    # Warmup (first jit call triggers compilation)
    for _ in range(4):
        batch_out = wte(params, input_ids)
        batch_out.block_until_ready()

    # Actual profiled runs
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        batch_out = wte(params, input_ids)
        batch_out.block_until_ready()
        times.append((time.perf_counter() - t0)*1000)
    times = jnp.array(times)
    print(f"et mean: {jnp.mean(times):.6f} ms")
    print(f"et med : {jnp.median(times):.6f} ms")
    print(f"et min : {jnp.min(times):.6f} ms")

    # See memory usage
    print_memory_stats(label="after")


# Function to see memory usage
def print_memory_stats(label: str = "",
                       ) -> None:
    """Quick util fun to print memory stats."""
    for device in jax.local_devices():
        print(f"Memory stats ({label}) {device}:")
        stats = device.memory_stats()
        if stats is None:
            print("  Not available...")
            continue
        peak = stats['peak_bytes_in_use'] / 1024**3
        live = stats['bytes_in_use'] / 1024**3
        lim = stats['bytes_limit'] / 1024**3
        print(f"  Live:  {live:.2f}/{lim:.2f} GB ({100*peak/lim:.2f}%)")
        print(f"  Peak:  {peak:.2f}/{lim:.2f} GB ({100*live/lim:.2f}%)")
