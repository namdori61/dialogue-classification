from typing import Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from modules import TransformerEmbeddings


@Model.register('token_transformer')
class TokenTransformerModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TransformerEmbeddings,
                 dialogue_encoder: TransformerEncoder,
                 discriminator: FeedForward,
                 dialogue_encoder_output_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self._text_field_embedder = text_field_embedder
        self._dialogue_encoder = dialogue_encoder
        self._discriminator = discriminator
        self._dialogue_encoder_output_dropout = (
            nn.Dropout(dialogue_encoder_output_dropout))

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
            text: tokens from a batch of sequences.
                From a `TextField`.
            is_buyer: boolean tensor that contain whether a token
                is sent by a buyer (1) or a seller (0).
            label: From a `LabelField`.

        Returns:
            An output dictionary consisting of:


        """
        # embedded_text: (num_sentences * num_tokens, batch_size, embedding_dim)
        embedded_text = self._text_field_embedder(text, is_buyer)
        embedded_text = embedded_text.permute(1, 0, 2)

        # attention_mask: (num_sentences * num_tokens, num_sentences * num_tokens)
        attention_mask = get_text_field_mask(text) == 0

        # dialogue_rep: (batch_size, dialogue_hidden_dim)
        dialogue_rep = self._dialogue_encoder(src=embedded_text,
                                              src_key_padding_mask=attention_mask)
        dialogue_rep = dialogue_rep.permute(1, 0, 2)
        dialogue_rep = dialogue_rep.mean(dim=1)
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
