"""
GPT-2 Stem (stages before the main Decoder).

This is just WTE + WPE followed by a Dropout.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_stem_wte import GPT2StemWTE
from models_x.gpt2.gpt2_stem_wpe import GPT2StemWPE

__all__ = ["GPT2Stem"]


@register_dataclass
@dataclass
class GPT2Stem():
    """
    GPT2 Stem (WTE + WPE and Dropout).
    """
    # Replaces __init__ since registered dataclass
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    wte: GPT2StemWTE = field(metadata=metadata)
    wpe: GPT2StemWPE = field(metadata=metadata)

    def init_params(self: Self,
                    prng_key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # Keys
        k1, k2 = jax.random.split(prng_key)

        return {'wte': self.wte.init_params(k1),
                'wpe': self.wpe.init_params(k2)}

    def dropout(self: Self,
                batch: Float[Array, "..."],
                p: float = 0.0,
                key: Array | None = None,
                deterministic: bool = True,
                ) -> Float[Array, "..."]:
        """
        Applies usual Dropout if training.
        Masks and scales to maintain the same expected value.
        """
        if 0.0 < p < 1.0 and isinstance(key, Array) and not deterministic:
            p_keep = 1.0 - p
            mask = jax.random.bernoulli(key=key,
                                        p=p_keep,
                                        shape=batch.shape)  # B x T x D
            batch = (batch * mask) / p_keep                 # B x T x D
        return batch

    def __call__(self: Self,
                 input_ids: Int[Array, "B T"],              # noqa: F722
                 params: dict[str, dict[str, Array]],
                 dropout_key: Array | None = None,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # Shape
        ntoks = input_ids.shape[-1]
        position_ids = jnp.arange(start=0,
                                  stop=ntoks,
                                  step=1,
                                  dtype=input_ids.dtype,
                                  device=input_ids.device)  # T

        # Embeddings
        tok_emb = self.wte(input_ids=input_ids,
                           params=params['wte'])            # B x T x D
        pos_emb = self.wpe(position_ids=position_ids,
                           params=params['wpe'])            # T x D
        batch = tok_emb + pos_emb                           # B x T x D

        # Dropout
        p = float(getattr(self.cfg, "p_drop_stem", 0.0))
        batch = self.dropout(batch=batch,
                             p=p,
                             key=dropout_key,
                             deterministic=deterministic)   # B x T x D

        return batch
