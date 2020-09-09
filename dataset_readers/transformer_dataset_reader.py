import json
import logging
from typing import Any, Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Field, Instance, Token, TokenIndexer
from allennlp.data.fields import (
    LabelField, TextField, SequenceLabelField
)
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('transformer')
class TransformerReader(DatasetReader):
    """
    Dataset reader for the custom dataset.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 maximum_length: int = 512,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = (
            token_indexers or {'tokens': SingleIdTokenIndexer()}
        )
        self._maximum_length = maximum_length

    @overrides
    def _read(self, file_path: str):
        # If `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info('Reading file at %s', file_path)

        with open(file_path) as dataset_file:
            dataset = dataset_file.readlines()

        logger.info('Reading the dataset')
        for line in dataset:
            obj = json.loads(line)
            messages = obj['messages']
            if 'label' in obj:
                if obj['label'] == 1:
                    label = 'fraud'
                else:
                    label = 'normal'
            else:
                label = None
            yield self.text_to_instance(messages=messages, label=label)

    @overrides
    def text_to_instance(self,
                         messages: List[Dict[str, Any]],
                         label: str = None
                         ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = []
        is_buyer_tags = []
        for message in messages:
            token_features = message['text']
            tokens += [Token(feature['token']) for feature in token_features]
            is_buyer_tags += [int(message['is_buyer'])] * len(token_features)
        if len(tokens) > self._maximum_length:
            tokens = tokens[:self._maximum_length]
            is_buyer_tags = is_buyer_tags[:self._maximum_length]
        text_field = TextField(tokens=tokens,
                               token_indexers=self._token_indexers)

        fields['text'] = text_field
        fields['is_buyer'] = SequenceLabelField(
            is_buyer_tags,
            sequence_field=text_field, label_namespace='is_buyer_tags')
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)
