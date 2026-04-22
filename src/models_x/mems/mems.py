"""
Memory Mosaics model main stages (not including task-specific heads).
This is just Stem and Decoder:
Input IDs --> [Stem] --> Word Embeddings --> [Decoder] --> Final Embeddings.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
from jaxtyping import Float, Int, Array
from models_x.mems.mems_config import MeMsConfig
from models_x.mems.mems_stem import MeMsStem
from models_x.mems.mems_decoder import MeMsDecoder

__all__ = ["MeMs"]


@register_dataclass
@dataclass(frozen=True)
class MeMs():
    """
    Memory Mosaics (Stem and Decoder).
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: MeMsConfig = field(metadata=metadata)
    stem: MeMsStem = field(metadata=metadata)
    decoder: MeMsDecoder = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: MeMsConfig,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Stem
        stem = MeMsStem.from_config(cfg=cfg, **kwargs)

        # Decoder
        decoder = MeMsDecoder.from_config(cfg=cfg, **kwargs)

        return cls(cfg=cfg, stem=stem, decoder=decoder)

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
                'decoder': self.decoder.init_params(key=key2)}

    def __call__(self: Self,
                 params: dict[str, Any],
                 input_ids: Int[Array, "B T"],              # noqa: F722
                 key: Array,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:                # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # PRNG keys
        key1, key2 = jax.random.split(key=key, num=2)

        # Word embeddings
        word_emb = self.stem(params=params['stem'],
                             input_ids=input_ids,
                             key=key1,
                             deterministic=deterministic)   # B x T x D

        # Final embeddings
        batch = self.decoder(params=params['decoder'],
                             arr=word_emb,
                             key=key2,
                             deterministic=deterministic)   # B x T x D

        return batch                                        # B x T x D
