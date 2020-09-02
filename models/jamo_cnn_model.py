from typing import Dict

import torch
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder

from allennlp.nn import InitializerApplicator, RegularizerApplicator


@Model.register('jamo_cnn')
class JamoCnnModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dialogue_encoder: nn.Module,
                 discriminator: FeedForward,
                 embedding_dropout: float = 0.0,
                 dialogue_encoder_output_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self._text_field_embedder = text_field_embedder
        self._dialogue_encoder = dialogue_encoder
        self._discriminator = discriminator
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._dialogue_encoder_output_dropout = (
            nn.Dropout(dialogue_encoder_output_dropout))

        # Size checks
        assert (
            text_field_embedder.get_output_dim()
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
                label: torch.IntTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs forward propagation.

        Args:
            text: tokens from a batch of sequences of sequences.
                From a `ListField([TextField])`.
            label: From a `LabelField`.

        Returns:
            An output dictionary consisting of:


        """
        # embedded_text: (batch_size, num_sentences, num_tokens, embedding_dim)
        embedded_text = self._text_field_embedder(text)
        embedded_text = self._embedding_dropout(embedded_text)

        # dialogue_rep: (batch_size, dialogue_hidden_dim)
        dialogue_rep = self._dialogue_encoder(embedded_text)
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
