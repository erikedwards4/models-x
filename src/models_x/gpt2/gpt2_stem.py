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
from models_x.fn.dropout import dropout
from models_x.nn.embedding import Embedding
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2Stem"]


@register_dataclass
@dataclass(frozen=True)
class GPT2Stem():
    """
    GPT2 Stem (WTE + WPE and dropout).
    """
    # Replaces __init__ since registered dataclass
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    wte: Embedding = field(metadata=metadata)
    wpe: Embedding = field(metadata=metadata)

    def __post_init__(self: Self,
                      ) -> None:
        """
        Standard dataclass method, called automatically.
        """
        # Word Token Embedding (WTE)
        wte = Embedding(num_embeddings=self.cfg.vocab_size,
                        embedding_dim=self.cfg.d_model,
                        init_std=self.cfg.init_std,
                        dtype=self.cfg.dtype)
        object.__setattr__(self, "wte", wte)

        # Word Position Embedding (WPE)
        wpe = Embedding(num_embeddings=self.cfg.n_positions,
                        embedding_dim=self.cfg.d_model,
                        init_std=self.cfg.init_std,
                        dtype=self.cfg.dtype)
        object.__setattr__(self, "wpe", wpe)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        key1, key2 = jax.random.split(key)

        # WTE
        params_wte = self.wte.init_params(key1)
        params_wpe = self.wte.init_params(key2)

        return {'wte': params_wte,
                'wpe': params_wpe}

    def __call__(self: Self,
                 params: dict[str, Any],
                 input_ids: Int[Array, "B T"],              # noqa: F722
                 key: Array | None = None,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # Position IDs
        position_ids = jnp.arange(start=0,
                                  stop=input_ids.shape[-1],
                                  step=1,
                                  dtype=input_ids.dtype,
                                  device=input_ids.device)  # T

        # Embeddings
        tok_emb = self.wte(params=params['wte'],
                           arr=input_ids)                   # B x T x D
        pos_emb = self.wpe(params=params['wpe'],
                           arr=position_ids)                # T x D
        batch = tok_emb + pos_emb                           # B x T x D

        # Dropout
        p = float(getattr(self.cfg, "p_drop_stem", 0.0))
        batch = dropout(arr=batch,
                        p=p,
                        key=key,
                        deterministic=deterministic)        # B x T x D

        return batch                                        # B x T x D
