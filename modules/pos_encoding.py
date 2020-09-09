import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    This code id made by pytorch transformer tutorial
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
    dropout : `float`, optional (default=`0.1`)
        This is dropout ratio to the sum of token embedding and positional
        encoding. The default value is 0.1.
    max_len : `int`, optional (default=`5000`)
        This is max length of positional encoding made. It would be expected
        that input token sentences are below mat length of it.
    """

    def __init__(self,
                 d_model: int = None,
                 dropout: float = 0.1,
                 max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)