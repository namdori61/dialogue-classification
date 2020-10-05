import json
import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class BertReader(Dataset):
    """
    Dataset reader for the Bert dataset.
    """

    def __init__(self,
                 file_path: str,
                 tokenizer: BertTokenizer = None,
                 maximum_length: int = 512,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self._maximum_length = maximum_length

        logger.info(f'Reading file at {file_path}')

        with open(file_path) as dataset_file:
            self.dataset = dataset_file.readlines()

    def text_to_instance(self,
                         messages: List[Dict[str, Any]],
                         label: int = None
                         ) -> Dict:
        processed_data = {}
        tokens = []
        is_buyer_tags = []
        for message in messages:
            token_features = message['text']
            tokens += [feature['token'] for feature in token_features]
            is_buyer_tags += [int(message['is_buyer'])] * len(token_features)
        if len(tokens) > self._maximum_length:
            is_buyer_tags = is_buyer_tags[:self._maximum_length]
        else:
            pad_num = self._maximum_length - len(is_buyer_tags)
            is_buyer_tags = is_buyer_tags + [0] * pad_num
        encoded_dict = self.tokenizer.encode_plus(text=' '.join(tokens),
                                                  add_special_tokens=True,
                                                  max_length=self._maximum_length,
                                                  truncation=True,
                                                  padding='max_length',
                                                  return_token_type_ids=False,
                                                  return_tensors='pt')

        processed_data['input_ids'] = encoded_dict['input_ids'].squeeze(0)
        processed_data['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)
        processed_data['is_buyer'] = torch.LongTensor(is_buyer_tags)
        if label is not None:
            processed_data['label'] = torch.LongTensor([label])

        return processed_data

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self,
                    idx: int = None) -> Dict:
        line = self.dataset[idx]
        obj = json.loads(line)
        messages = obj['messages']
        if 'label' in obj:
            label = obj['label']
        else:
            label = None
        return self.text_to_instance(messages=messages, label=label)