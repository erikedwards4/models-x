"""
Pytest function for gpt2/gpt2_stem_wte.py.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
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

    # Get config
    config = GPT2Config(vocab_size=vocab_size,
                        d_model=d_model,
                        dtype=dtype)

    # Get mdl
    wte = GPT2StemWTE(config=config)
    assert isinstance(wte, GPT2StemWTE)
    assert callable(wte)
    assert wte.dtype == config.dtype

    # Get PRNG key
    prng_key = jax.random.PRNGKey(seed=0)

    # Get params dict
    params = wte.init_params(prng_key=prng_key)
    assert 'wte' in params
    assert isinstance(params['wte'], Array)
    assert params['wte'].dtype == jnp.dtype(config.dtype)
    assert jnp.std(params['wte']).item() > 0.0

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = (nbatch, ntoks)
    # device = "cuda"
    input_ids = jax.random.randint(key=prng_key,
                                   shape=size_in,
                                   minval=0,
                                   maxval=vocab_size,
                                   dtype=jnp.int32)

    # Test __call__
    batch_out = wte(params=params,
                    input_ids=input_ids)
    print(f"batch_out.dtype = {batch_out.dtype}")
    print(f"batch_out.shape = {batch_out.shape}")
    assert isinstance(batch_out, Float[jnp.ndarray, "..."])
    assert batch_out.dtype == jnp.dtype(config.dtype)
    assert batch_out.shape == (nbatch, ntoks, wte.d_model)
    assert jnp.all(jnp.isfinite(batch_out))

    # Test embedding look-up
    embd0 = params['wte'][input_ids[0, 0]]
    assert jnp.allclose(batch_out[0, 0], embd0)

    # # Set for inference
    # for p in wte.parameters():
    #     p.requires_grad = False
    # wte.eval()

    # # Profile inference
    # activities = [torch.profiler.ProfilerActivity.CPU,
    #               torch.profiler.ProfilerActivity.CUDA]
    # wait, warmup, active, repeat, buffer = 6, 6, 10, 2, 6
    # its = (wait + warmup + active) * repeat + buffer
    # schedule = torch.profiler.schedule(wait=wait, warmup=warmup,
    #                                    active=active, repeat=repeat)
    # iprofile = torch.profiler.profile(activities=activities,
    #                                   schedule=schedule,
    #                                   record_shapes=False,
    #                                   with_stack=True)
    # with iprofile as iprof:
    #     with torch.amp.autocast(device_type=device,
    #                             dtype=amp_dtype):
    #         with torch.inference_mode():
    #             for _ in range(its):
    #                 with torch.profiler.record_function("mdl_inference"):
    #                     _ = wte(batch_in)
    #                 iprof.step()
    # print(iprof.key_averages().table(sort_by="cpu_time_total",
    #                                  row_limit=6))
    # print(f"nits = {its}")

    # # Profile GPU memory usage
    # gpu_properties = torch.cuda.get_device_properties(0)
    # gb_tot = gpu_properties.total_memory / 1024**3
    # print(f"GPU type: {torch.cuda.get_device_name(0)}")
    # print(f"GPU total VRAM = {gb_tot:.2f} GiB")

    # # See model GPU memory usage
    # gb_buffs = sum(b.nelement()*b.element_size()
    #                for b in wte.buffers()) / 1024**3
    # gb_params = sum(p.nelement()*p.element_size()
    #                 for p in wte.parameters()) / 1024**3
    # gb_model = gb_buffs + gb_params
    # print(f"GPU VRAM usage (buffs) : {gb_buffs:.2f} GiB / {gb_tot:.2f} GiB")
    # print(f"GPU VRAM usage (params): {gb_params:.2f} GiB / {gb_tot:.2f} GiB")
    # print(f"GPU VRAM usage (model) : {gb_model:.2f} GiB / {gb_tot:.2f} GiB")

    # # See peak GPU memory usage
    # gb_peak = torch.cuda.max_memory_allocated() / 1024**3
    # use_percent = 100 * (gb_peak / gb_tot)
    # print(f"Peak GPU Memory Usage: {gb_peak:.2f} GiB / {gb_tot:.2f} GiB")
    # print(f"Peak GPU Memory Usage (%): {use_percent:.2f}%")
