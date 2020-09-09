from typing import Dict, Union

import torch
import torch.nn as nn

from allennlp.data import TextFieldTensors
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from modules import positional_encoding


class TransformerEmbeddings(nn.Module):
    """
    This code id inspired by huggingface transformer BertEmbeddings class
    (https://github.com/huggingface/transformers).
    Transformer embeddings for Transformer input token sequence.
    It inject some information about the relative or absolute position
    of the tokens in the sequence.
    We add "positional embeddings" to the input embeddings at the bottoms
    of the encoder and decoder stacks.
    The positional embeddings have the same dimension d_model as the
    embeddings, so that the two can be summed.
    Also, we add "is buyer embeddings" to the input embeddings at the
    bottoms of the encoder and decoder stacks to distinguish tokens from
    buyer or seller.
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
                 vocab_size: int = None,
                 embedding_dim: int = None,
                 dropout: float = 0.1) -> None:
        super(TransformerEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_embeddings = BasicTextFieldEmbedder(
            token_embedders={
                'tokens': Embedding(embedding_dim=self.embedding_dim,
                                    num_embeddings=vocab_size)
            }
        )
        self.is_buyer_embeddings = Embedding(embedding_dim=self.embedding_dim,
                                             num_embeddings=2)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)

    def forward(self,
                input_ids: TextFieldTensors = None,
                is_buyer_tags: torch.IntTensor = None) -> torch.Tensor:
        input_shape = tuple(input_ids['tokens']['tokens'].size())
        seq_length = input_shape[1]

        if is_buyer_tags is None:
            is_buyer_tags = torch.zeros(input_shape, dtype=torch.long)

        input_embeddings = self.input_embeddings(input_ids)
        position_embeddings = positional_encoding(d_model=self.embedding_dim,
                                                  max_len=seq_length)
        position_embeddings = position_embeddings.to(input_embeddings.device)
        is_buyer_embeddings = self.is_buyer_embeddings(is_buyer_tags)

        embeddings = input_embeddings + position_embeddings + is_buyer_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
