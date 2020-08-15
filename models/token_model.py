from typing import Dict

import torch
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Embedding, FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask


@Model.register('token')
class TokenModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 dialogue_encoder: Seq2VecEncoder,
                 discriminator: FeedForward,
                 is_buyer_embedding_dim: int,
                 embedding_dropout: float = 0.0,
                 sentence_encoder_output_dropout: float = 0.0,
                 dialogue_encoder_output_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self._text_field_embedder = text_field_embedder
        self._sentence_encoder = TimeDistributed(sentence_encoder)
        self._dialogue_encoder = dialogue_encoder
        self._discriminator = discriminator
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._sentence_encoder_output_dropout = (
            nn.Dropout(sentence_encoder_output_dropout))
        self._dialogue_encoder_output_dropout = (
            nn.Dropout(dialogue_encoder_output_dropout))

        self._is_buyer_embedding = Embedding(
            embedding_dim=is_buyer_embedding_dim,
            num_embeddings=2
        )

        # Size checks
        assert (
            text_field_embedder.get_output_dim()
            == sentence_encoder.get_input_dim()
        )
        assert (
            sentence_encoder.get_output_dim() + is_buyer_embedding_dim
            == dialogue_encoder.get_input_dim()
        )
        assert (
            dialogue_encoder.get_output_dim()
            == discriminator.get_input_dim()
        )

        self.metrics = {
            'accuracy': CategoricalAccuracy()
        }

        initializer(self)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset=reset)
                   for k, v in self.metrics.items()}
        return metrics

    def forward(self,
                text: TextFieldTensors,
                is_buyer: torch.IntTensor,
                label: torch.IntTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs forward propagation.

        Args:
            text: tokens from a batch of sequences of sequences.
                From a `ListField([TextField])`.
            is_buyer: boolean values that contain whether a sentence
                is sent by a buyer (1) or a seller (0).
            label: From a `LabelField`.

        Returns:
            An output dictionary consisting of:


        """
        # embedded_text: (batch_size, num_sentences, num_tokens, embedding_dim)
        embedded_text = self._text_field_embedder(text)
        embedded_text = self._embedding_dropout(embedded_text)

        # sentence_mask: (batch_size, num_sentences, num_tokens)
        sentence_mask = get_text_field_mask(text, num_wrapping_dims=1)
        # dialogue_mask: (batch_size, num_sentences)
        dialogue_mask = sentence_mask.any(dim=2)

        # sentence_reps: (batch_size, num_sentences, sentence_hidden_dim)
        sentence_reps = self._sentence_encoder(embedded_text,
                                               mask=sentence_mask)
        sentence_reps = self._sentence_encoder_output_dropout(sentence_reps)

        # is_buyer_embs: (batch_size, num_sentences, is_buyer_embedding_dim)
        is_buyer_embs = self._is_buyer_embedding(is_buyer)
        dial_enc_inputs = torch.cat([sentence_reps, is_buyer_embs], dim=2)
        # dialogue_rep: (batch_size, dialogue_hidden_dim)
        dialogue_rep = self._dialogue_encoder(dial_enc_inputs,
                                              mask=dialogue_mask)
        dialogue_rep = self._dialogue_encoder_output_dropout(dialogue_rep)

        logits = self._discriminator(dialogue_rep)

        output_dict = {
            'logits': logits
        }
        if label is not None:
            loss = nn.functional.cross_entropy(input=logits, target=label)
            self.metrics['accuracy'](predictions=logits, gold_labels=label)
            output_dict['loss'] = loss

        return output_dict
