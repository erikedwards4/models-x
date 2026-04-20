"""
GPT-2 WTE (Word Token Embedding) part of the stem.

This is a look-up from integer token IDs in [0, vocab_size)
to float embedding vectors with length d_model.
"""

from typing import Self
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2StemWTE"]


@register_dataclass
@dataclass
class GPT2StemWTE():
    """
    GPT2 WTE (Word Token Embedding).
    """
    # Replaces __init__ since registered dataclass
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)

    # def __init__(self: Self,
    #              config: GPT2Config,
    #              ) -> None:
    #     """
    #     vocab_size: size of vocab --> num_embeddings learned
    #                 inputs to forward are ints in [0, vocab_size)
    #     d_model: model dimension --> embedding_dim for learned vecs
    #     dtype: JAX dtype
    #     """
    #     # Self attributes
    #     self.vocab_size = int(config.vocab_size)
    #     self.d_model = int(config.d_model)
    #     self.dtype: jax.typing.DTypeLike = config.dtype

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Embedding weights shape
        shape = (int(self.cfg.vocab_size),
                 int(self.cfg.d_model))

        # GPT-2 uses a normal dist with std = 0.02
        std = float(self.cfg.init_std)

        # Embedding weights
        wte = jax.random.normal(key=key,
                                shape=shape,
                                dtype=self.cfg.dtype,
                                ) * std             # V x D

        return {'wte': wte}

    def __call__(self: Self,
                 params: dict[str, Array],
                 input_ids: Int[Array, "B T"],      # noqa: F722
                 ) -> Float[Array, "B T D"]:        # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # torch.nn.Embedding lookup --> jnp.take
        return jnp.take(a=params['wte'],
                        indices=input_ids,
                        axis=0)                     # B x T x D
