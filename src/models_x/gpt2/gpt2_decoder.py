"""
GPT-2 Transformer Decoder stage,
which is the main section with 1-12 decoder layers.

See gpt2_decoder_block.py for the MHA (multi-head attention).
This Decoder is simply a stack of these blocks, e.g.
12 layers (repeated blocks) for the GPT-2 small version.

The input is the word embeddings from GPT2Stem with dim d_model.
The output is the final contextualized embeddings with dim d_model.
"""

from typing import Self, Any
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from models_x.nn.layer_norm import LayerNorm
from models_x.gpt2.gpt2_config import GPT2Config
from models_x.gpt2.gpt2_decoder_block import GPT2DecoderBlock

__all__ = ["GPT2Decoder"]


@register_dataclass
@dataclass(frozen=True)
class GPT2Decoder():
    """
    GPT-2 Transformer Decoder.
    """
    # __init__
    metadata = dict(static=True)    # pylint: disable=use-dict-literal
    cfg: GPT2Config = field(metadata=metadata)
    blocks: tuple[GPT2DecoderBlock, ...] = field(metadata=metadata)
    lnorm_f: LayerNorm = field(metadata=metadata)

    @classmethod
    def from_config(cls: type[Self],
                    cfg: GPT2Config,
                    **kwargs: Any,
                    ) -> Self:
        """
        Instantiate from_config, with optional kwargs to override.
        """
        # Config attributes
        nblocks = int(kwargs.get('nblocks', cfg.nblocks))
        d_model = int(kwargs.get('d_model', cfg.d_model))
        lnorm_eps = float(kwargs.get('lnorm_eps', cfg.lnorm_eps))
        dtype = jnp.dtype(kwargs.get('dtype', cfg.dtype))

        # List of decoder blocks
        blocks: list[GPT2DecoderBlock] = []
        for _ in range(nblocks):
            block = GPT2DecoderBlock.from_config(cfg=cfg, **kwargs)
            blocks.append(block)

        # Final layer norm
        lnorm_f = LayerNorm(normalized_shape=d_model,
                            eps=lnorm_eps,
                            bias=True,
                            dtype=dtype)

        return cls(cfg=cfg, blocks=tuple(blocks), lnorm_f=lnorm_f)

    def init_params(self: Self,
                    key: Array,
                    ) -> dict[str, Any]:
        """
        Initialize the parameters dict.
        """
        # PRNG keys
        keys = jax.random.split(key=key, num=len(self.blocks))

        # Params dict
        params: dict[str, Any] = {}

        # Decoder blocks
        for b, block in enumerate(self.blocks):
            params[f'block{b}'] = block.init_params(keys[b])

        # Final LayerNorm
        params['lnorm_f'] = self.lnorm_f.init_params()

        return params

    def __call__(self: Self,
                 params: dict[str, Any],
                 arr: Float[Array, "B T D"],            # noqa: F722
                 key: Array,
                 deterministic: bool = True,
                 ) -> Float[Array, "B T D"]:            # noqa: F722
        """
        B = batch_size (or micro-batch size)
        T = ntoks (num tokens, input seq len)
        D = d_model (num embeddings, model dim)
        """
        # PRNG keys
        keys = jax.random.split(key=key, num=len(self.blocks))

        # Main Decoder blocks
        for b, block in enumerate(self.blocks):
            arr = block(params=params[f'block{b}'],
                        arr=arr,
                        key=keys[b],
                        deterministic=deterministic)    # B x T x D

        # Final LayerNorm
        arr = self.lnorm_f(params=params['lnorm_f'],
                           arr=arr)                     # B x T x D

        return arr                                      # B x T x D
