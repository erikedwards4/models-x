"""
GPT-2 WPE (Word Position Embedding) part of the stem.

This is a look-up from integer position IDs in [0, n_positions)
to float embedding vectors with length d_model.
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2StemWPE"]


@register_dataclass
@dataclass
class GPT2StemWPE():
    """
    GPT2 WPE (Word Position Embedding).
    """
    # Replaces __init__ since registered dataclass
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Embedding weights shape
        shape = (int(self.cfg.n_positions),
                 int(self.cfg.d_model))

        # GPT-2 uses a normal dist with std = 0.02
        std = float(self.cfg.init_std)

        # Embedding weights
        wpe = jax.random.normal(key=key,
                                shape=shape,
                                dtype=self.cfg.dtype,
                                ) * std             # P x D

        return {'wpe': wpe}

    def __call__(self: Self,
                 params: dict[str, Array],
                 position_ids: Int[Array, "T"],     # noqa: F722
                 ) -> Float[Array, "T D"]:          # noqa: F722
        """
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # torch.nn.Embedding lookup --> jnp.take
        return jnp.take(a=params['wpe'],
                        indices=position_ids,
                        axis=0)                     # T x D
