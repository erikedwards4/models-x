"""
Pytest function for mems/mems_config.py.
"""
from dataclasses import is_dataclass
import pytest
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from models_x.mems.mems_config import MeMsConfig


# mems_config.MeMsConfig
@pytest.mark.parametrize("vocab_size", (50257, ))
@pytest.mark.parametrize("d_model", (768, ))
@pytest.mark.parametrize("dtype", (jnp.float32, ))
def test_mems_config(vocab_size, d_model, dtype):
    """
    Pytest mems_config.MeMsConfig.
    """
    print("")

    # Get config
    cfg = MeMsConfig(vocab_size=vocab_size,
                     d_model=d_model,
                     dtype=dtype)
    assert is_dataclass(cfg)
    assert hasattr(cfg, 'vocab_size')
    assert cfg.vocab_size == vocab_size

    # Flatten the object
    leaves, treedef = tree_flatten(tree=cfg)
    # print(f"Aux_data (treedef): {treedef}, {type(treedef)}")
    # print(f"Children (leaves): {leaves}, {type(leaves)}")
    # Expected: [] (since everything is marked static=True)

    # Check leaves (empty since only static in config)
    assert isinstance(leaves, list)
    assert len(leaves) == 0

    # Check treedef
    assert isinstance(treedef, PyTreeDef)
    metadata = treedef.node_data()
    print(f"Metadata: {metadata}, {type(metadata)}")
    assert isinstance(metadata, tuple)
    assert len(metadata) > 0
    assert hash(treedef) != 0
    # If the hash fails, one of the 'static=True'
    # fields is not a valid JAX constant

    # Unflatten the object (reconstruction)
    new_cfg = tree_unflatten(treedef=treedef,
                             leaves=leaves)
    assert new_cfg == cfg
    print("Pytree reconstruction successful!")

    # Test JIT compatibility
    @jax.jit
    def get_d_model(cfg: MeMsConfig,
                    ) -> int:
        return cfg.d_model
    jit_d_model = get_d_model(cfg=cfg)
    print(f"JIT output: d_model = {jit_d_model}")
