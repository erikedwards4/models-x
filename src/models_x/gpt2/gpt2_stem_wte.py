"""
GPT-2 WTE (Word Token Embedding) part of the stem.

This is a look-up from integer token IDs in [0, vocab_size)
to float embedding vectors with length d_model.
"""

from typing import Self
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from models_x.gpt2.gpt2_config import GPT2Config

__all__ = ["GPT2StemWTE"]


class GPT2StemWTE():
    """
    GPT2 WTE (Word Token Embedding).
    """
    def __init__(self: Self,
                 config: GPT2Config,
                 ) -> None:
        """
        vocab_size: size of vocab --> num_embeddings learned
                    inputs to forward are ints in [0, vocab_size)
        d_model: model dimension --> embedding_dim for learned vecs
        dtype: JAX dtype
        """
        # Self attributes
        self.vocab_size = int(config.vocab_size)
        self.d_model = int(config.d_model)
        self.dtype: jax.typing.DTypeLike = config.dtype

        # Embedding weights shape
        self.wte_shape = (self.vocab_size, self.d_model)

    def init_params(self: Self,
                    prng_key: Array,
                    ) -> dict[str, Array]:
        """
        Initialize the parameters dict.
        """
        # Embedding weights
        # GPT-2 uses a normal distribution with std = 0.02
        wte = jax.random.normal(key=prng_key,
                                shape=self.wte_shape,
                                dtype=self.dtype)   # V x D
        wte = wte * 0.02                            # V x D

        return {'wte': wte}

    def __call__(self: Self,
                 params: dict[str, Array],
                 input_ids: Int[Array, "B T"],      # noqa: F722
                 ) -> Float[Array, "B T D"]:        # noqa: F722
        """
        B = batch_size
        T = ntoks (num tokens, input seq len)
        D = d_model (model dim)
        """
        # torch.nn.Embedding lookup --> jnp.take
        return jnp.take(a=params['wte'],
                        indices=input_ids,
                        axis=0)                     # B x T x D
