"""
Pytest function for gpt2/gpt2_lm_head.py.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.util.profile_callable import profile_callable
from models_x.util.print_memory_stats import print_memory_stats
from models_x.gpt2.gpt2_lm_head import GPT2LMHead


# gpt2_lm_head.GPT2LMHead
def test_gpt2_lm_head():
    """
    Pytest gpt2_lm_head.GPT2LMHead.
    """
    # Start
    print("")
    device = jax.devices("gpu")[0]
    print_memory_stats(label="start")

    # Get mdl
    mdl = GPT2LMHead.from_config()
    assert isinstance(mdl, GPT2LMHead)
    assert callable(mdl)

    # PRNG keys
    prng_key = jax.random.PRNGKey(seed=0)
    data_key, wte_key, call_key = \
        jax.random.split(key=prng_key, num=3)

    # Get params dict
    params = mdl.init_params()
    assert isinstance(params, dict)

    # Check sample_lear (should be empty)
    params = jax.device_put(x=params, device=device)
    sample_leaf = jax.tree_util.tree_leaves(params)
    print(f"sample_leaf = {sample_leaf}")

    # Make input data
    nbatch = 4          # micro-batch size
    ntoks = 1024
    d_model = 768
    vocab_size = 50257
    size_in = (nbatch, ntoks, d_model)
    size_wte = (vocab_size, d_model)
    batch_in = jax.random.normal(key=data_key,
                                 shape=size_in,
                                 dtype=jnp.float32,
                                 ).to_device(device)
    wte = jax.random.normal(key=wte_key,
                            shape=size_wte,
                            dtype=jnp.float32,
                            ).to_device(device)

    # Test __call__
    logits = mdl(params=params,
                 arr=batch_in,
                 key=call_key,
                 wte=wte)
    print(f"logits.dtype = {logits.dtype}")
    print(f"logits.shape = {logits.shape}")
    assert isinstance(logits, Float[Array, "..."])
    assert logits.dtype == batch_in.dtype
    assert logits.device == batch_in.device
    assert logits.shape == (nbatch, ntoks, vocab_size)
    assert jnp.all(jnp.isfinite(logits))

    # JIT compile
    mdl_jit = jax.jit(mdl)
    logits = mdl_jit(params=params,
                     arr=batch_in,
                     key=call_key,
                     wte=wte)

    # See memory usage
    print_memory_stats(label="after")

    # Profile
    profile_callable(fun=mdl_jit,
                     n_runs=32,
                     params=params,
                     arr=batch_in,
                     key=call_key,
                     wte=wte)
