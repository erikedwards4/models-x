"""
GPT-2 model LM (language model) head.

This is the most-used, canonical head for GPT-2,
used in pre-training and fine-tuning, and for text
generation and completion. This is for the general
task of autoregressive next-token prediction, e.g.:
Loss = -Sum[ log(P(x_t | x_<t)) ]

GPT-2 uses weight tying -- the LM head weight is
the WTE matrix transposed, so there are 0 params
here, and the WTE weights (embedding vecs) must be
input to the __call__ method for the forward pass.

Returns B x T x V logits (raw log probs) for each
batch, input token, and vocab item. The softmax of
the output across the vocab dimension would be the
posterior probs. That is, the probability that the
token is the vocab item in [0, V) given the past.
"""

from typing import Self, Any
from dataclasses import dataclass
from jax.tree_util import register_dataclass
import jax
from jaxtyping import Array, Float

__all__ = ["GPT2LMHead"]


@register_dataclass
@dataclass(frozen=True)
class GPT2LMHead():
    """
    GPT-2 LM Head for autoregressive next-token prediction.
    """
    # __init__

    @classmethod
    def from_config(cls: type[Self],
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        return cls()

    def init_params(self: Self,
                    _key: Array | None = None,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        This is empty because reusing the WTE params.
        """
        return {}

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],                # noqa: F722
                 wte: Float[Array, "V D"],                  # noqa: F722
                 key: Array | None = None,
                 ) -> Float[Array, "B T V"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (hiden size, model dim)
        V = vocab_size (WTE num_embeddings)
        """
        # Linear projection: hidden states --> logits
        # logits = jnp.dot(arr, wte.T)                      # B x T x V
        # logits = jnp.einsum("btd,vd->btv", arr, wte)      # B x T x V

        # Logits (a bit faster, fused @ and .T)
        ds = (((2,), (1,)), ((), ()))
        logits = jax.lax.dot_general(lhs=arr,
                                     rhs=wte,
                                     dimension_numbers=ds)  # B x T x V

        return logits                                       # B x T x V
