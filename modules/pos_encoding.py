import math

import torch
import torch.nn as nn


def positional_encoding(d_model: int = None,
                        max_len: int = 512) -> torch.Tensor:
    """
    This code id inspired by pytorch transformer tutorial
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html).
    Positional Encoding for Transformer input token sequence.
    It inject some information about the relative or absolute position
    of the tokens in the sequence.
    We add "positional encodings" to the input embeddings at the bottoms
    of the encoder and decoder stacks.
    The positional encodings have the same dimension d_model as the
    embeddings, so that the two can be summed.
    # Parameters
    d_model : `int`, required
        This is token embedding dimension.
    max_len : `int`, optional (default=`512`)
        This is max length of positional encoding made. It would be expected
        that input token sentences are below mat length of it.
    """

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(start=0,
                            end=max_len,
                            dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(start=0,
                                      end=d_model,
                                      step=2,
                                      dtype=torch.float) *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
