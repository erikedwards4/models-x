"""
GPT-2 model with the default, canonical head,
which is the LM Head for next-token prediction.

This is: Stem, Decoder, Head:
Token IDs --> [Stem] --> Word Embeddings --> [Decoder]
--> Contextualized Embeddings --> [Head] --> Logits
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
from jaxtyping import Array, Float, Int
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_stem import GPT2Stem
from models_x.gpt2.gpt2_decoder import GPT2Decoder
from models_x.gpt2.gpt2_lm_head import GPT2LMHead

__all__ = ["GPT2LM"]


@register_dataclass
@dataclass(frozen=True)
class GPT2LM():
    """
    GPT-2 LM (Stem and Decoder and LM Head).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    stem: GPT2Stem = field(metadata=metadata)
    decoder: GPT2Decoder = field(metadata=metadata)
    head: GPT2LMHead = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: GPT2Config,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Stem
        stem = GPT2Stem.from_config(cfg=cfg, **kwargs)

        # Decoder
        decoder = GPT2Decoder.from_config(cfg=cfg, **kwargs)

        # Head
        head = GPT2LMHead.from_config()

        return cls(cfg=cfg, stem=stem, decoder=decoder, head=head)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Params dict
        return {'stem': self.stem.init_params(key=key1),
                'decoder': self.decoder.init_params(key=key2),
                'head': self.head.init_params()}

    def __call__(self: Self,
                 params: dict[str, Any],
                 token_ids: Int[Array, "B T"],              # noqa: F722
                 key: Array,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T V"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        V = vocab_size (WTE num_embeddings)
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Stem --> word embeddings
        word_emb = self.stem(params=params['stem'],
                             token_ids=token_ids,
                             key=key1,
                             deterministic=deterministic)       # B x T x D

        # Decoder --> final embeddings
        final_emb = self.decoder(params=params['decoder'],
                                 arr=word_emb,
                                 key=key2,
                                 deterministic=deterministic)   # B x T x D

        # LM Head --> logits
        logits = self.head(params=params['head'],
                           arr=final_emb,
                           wte=params['stem']['wte']['w'])      # B x T x V

        return logits                                           # B x T x V
