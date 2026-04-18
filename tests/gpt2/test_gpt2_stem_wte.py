"""
Pytest function for gpt2/gpt2_stem_wte.py.
"""
from pydantic import BaseModel
import pytest
import torch
from models_ext.gpt2.gpt2_stem_wte import GPT2StemWTE, GPT2StemWTECfg


# gpt2_stem_wte.GPT2StemWTE
@pytest.mark.parametrize("vocab_size", (50257, ))
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (torch.float32, ))
@pytest.mark.parametrize("device", ("cuda", ))
def test_gpt2_stem_wte(vocab_size, d_model, dtype, device):
    """
    Pytest gpt2_stem_wte.GPT2StemWTE.
    """
    # Make random data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    size_in = [nbatch, ntoks]
    batch_in = torch.randint(low=0,
                             high=vocab_size,
                             size=size_in,
                             dtype=torch.long,
                             device=device,
                             requires_grad=False)

    # Get mdl
    wte = GPT2StemWTE(vocab_size=vocab_size,
                      d_model=d_model,
                      dtype=dtype,
                      device=device)

    # Basic checks
    assert isinstance(wte, torch.nn.Module)
    assert isinstance(wte, GPT2StemWTE)
    assert callable(wte)

    # AMP dtype
    amp_dtype = torch.float16

    # Test block forward
    with torch.amp.autocast(device_type=device,
                            dtype=amp_dtype):
        batch_out = wte(batch_in)
    assert batch_out.ndim == batch_in.ndim + 1
    assert batch_out.size(0) == batch_in.size(0) == nbatch
    assert batch_out.size(-1) == wte.d_model == d_model
    assert batch_out.size(-2) == batch_in.size(-1) == ntoks
    assert torch.all(torch.isfinite(batch_out))

    # Verbose
    print("")
    print(f"batch_out.shape = {batch_out.shape}")

    # Set for inference
    for p in wte.parameters():
        p.requires_grad = False
    wte.eval()

    # Profile inference
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    wait, warmup, active, repeat, buffer = 6, 6, 10, 2, 6
    its = (wait + warmup + active) * repeat + buffer
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup,
                                       active=active, repeat=repeat)
    iprofile = torch.profiler.profile(activities=activities,
                                      schedule=schedule,
                                      record_shapes=False,
                                      with_stack=True)
    with iprofile as iprof:
        with torch.amp.autocast(device_type=device,
                                dtype=amp_dtype):
            with torch.inference_mode():
                for _ in range(its):
                    with torch.profiler.record_function("mdl_inference"):
                        _ = wte(batch_in)
                    iprof.step()
    print(iprof.key_averages().table(sort_by="cpu_time_total",
                                     row_limit=6))
    print(f"nits = {its}")

    # Profile GPU memory usage
    gpu_properties = torch.cuda.get_device_properties(0)
    gb_tot = gpu_properties.total_memory / 1024**3
    print(f"GPU type: {torch.cuda.get_device_name(0)}")
    print(f"GPU total VRAM = {gb_tot:.2f} GiB")

    # See model GPU memory usage
    gb_buffs = sum(b.nelement()*b.element_size()
                   for b in wte.buffers()) / 1024**3
    gb_params = sum(p.nelement()*p.element_size()
                    for p in wte.parameters()) / 1024**3
    gb_model = gb_buffs + gb_params
    print(f"GPU VRAM usage (buffs) : {gb_buffs:.2f} GiB / {gb_tot:.2f} GiB")
    print(f"GPU VRAM usage (params): {gb_params:.2f} GiB / {gb_tot:.2f} GiB")
    print(f"GPU VRAM usage (model) : {gb_model:.2f} GiB / {gb_tot:.2f} GiB")

    # See peak GPU memory usage
    gb_peak = torch.cuda.max_memory_allocated() / 1024**3
    use_percent = 100 * (gb_peak / gb_tot)
    print(f"Peak GPU Memory Usage: {gb_peak:.2f} GiB / {gb_tot:.2f} GiB")
    print(f"Peak GPU Memory Usage (%): {use_percent:.2f}%")

    # Test config
    print("")
    cfg = GPT2StemWTECfg.model_validate(obj=wte,
                                        strict=True,
                                        from_attributes=True)
    print(f"type(cfg) = {type(cfg)}\ncfg = {cfg}")
    assert isinstance(cfg, BaseModel)
    assert isinstance(cfg, GPT2StemWTECfg)
    wte = GPT2StemWTE.from_cfg(cfg=cfg)
    assert isinstance(wte, GPT2StemWTE)
