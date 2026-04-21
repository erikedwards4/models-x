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
    GPT-2 Stem (WTE + WPE and dropout).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    wte: Embedding = field(metadata=metadata)
    wpe: Embedding = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: GPT2Config,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Config attributes
        vocab_size = int(kwargs.get('vocab_size', cfg.vocab_size))
        n_positions = int(kwargs.get('n_positions', cfg.n_positions))
        d_model = int(kwargs.get('d_model', cfg.d_model))
        init_std = float(kwargs.get('init_std', cfg.init_std))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # Word Token Embedding (WTE)
        wte = Embedding(num_embeddings=vocab_size,
                        embedding_dim=d_model,
                        init_std=init_std,
                        dtype=dtype)

        # Word Position Embedding (WPE)
        wpe = Embedding(num_embeddings=n_positions,
                        embedding_dim=d_model,
                        init_std=init_std,
                        dtype=dtype)

        return cls(cfg=cfg, wte=wte, wpe=wpe)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Params dict
        return {'wte': self.wte.init_params(key=key1),
                'wpe': self.wpe.init_params(key=key2)}

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
                                  dtype=jnp.int32)          # T

        # Embeddings
        tok_emb = self.wte(params=params['wte'],
                           arr=input_ids)                   # B x T x D
        pos_emb = self.wpe(params=params['wpe'],
                           arr=position_ids)                # T x D
        word_emb = tok_emb + pos_emb                        # B x T x D

        # Dropout
        p = 0.0 if deterministic else \
            float(getattr(self.cfg, 'p_drop_stem', 0.0))
        word_emb = dropout(arr=word_emb,
                           p=p,
                           key=key)                         # B x T x D

        return word_emb                                     # B x T x D
