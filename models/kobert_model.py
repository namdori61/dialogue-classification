from typing import Any, Tuple, Dict, Union, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional.classification import accuracy, precision, recall, f1_score, auroc
from transformers.modeling_bert import BertModel
from transformers import AdamW

from dataset_readers import KoBertReader


class TokenKoBertModel(LightningModule):
    def __init__(self,
                 model: BertModel = None,
                 tokenizer: Any = None,
                 train_path: str = None,
                 dev_path: str = None,
                 test_path: str = None,
                 maximum_length: int = 512,
                 num_classes: int = 2,
                 cuda_device: int = 0,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 lr: float = 2e-5,
                 weight_decay: float = 0.1,
                 warm_up: int = 500):
        super(TokenKoBertModel, self).__init__()

        self.num_classes = num_classes
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up = warm_up

        if train_path is not None:
            self.train_dataset = KoBertReader(file_path=train_path,
                                              tokenizer=tokenizer,
                                              maximum_length=maximum_length)
        if dev_path is not None:
            self.dev_dataset = KoBertReader(file_path=dev_path,
                                            tokenizer=tokenizer,
                                            maximum_length=maximum_length)
        if test_path is not None:
            self.test_dataset = KoBertReader(file_path=test_path,
                                             tokenizer=tokenizer,
                                             maximum_length=maximum_length)

        self.save_hyperparameters()

        self.text_embedding = model

        self.classifier = nn.Linear(self.text_embedding.config.hidden_size, self.num_classes)

    def forward(self,
                batch: Dict = None) -> float:
        text_embedded = self.text_embedding(batch['input_ids'],
                                            token_type_ids=batch['is_buyer'],
                                            attention_mask=batch['attention_mask'])
        text_embedded = text_embedded[1]

        logits = self.classifier(text_embedded)

        return logits

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.cuda_device > 0:
            sampler = DistributedSampler(self.train_dataset)
        else:
            sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=sampler,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        sampler = SequentialSampler(self.dev_dataset)
        val_dataloader = DataLoader(self.dev_dataset,
                                    sampler=sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        sampler = SequentialSampler(self.test_dataset)
        test_dataloader = DataLoader(self.test_dataset,
                                     sampler=sampler,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers)
        return test_dataloader

    def configure_optimizers(self) -> Optional[
        Union[
            Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
        ]
    ]:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr,
                          eps=1e-8)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.warm_up:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warm_up))
        else:
            lr_scale = min(1., float(self.warm_up) / float(self.trainer.global_step + 1))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self,
                      batch: Dict = None,
                      batch_idx: int = None) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        pred = torch.argmax(logits, dim=1)

        acc = accuracy(pred, labels.view(-1), num_classes=self.num_classes)

        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', acc, on_epoch=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self,
                        batch: Dict = None,
                        batch_idx: int = None) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        pred = torch.argmax(logits, dim=1)

        self.log('val_loss', loss)

        return {'loss': loss, 'pred': pred, 'label': labels}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([accuracy(x['pred'], x['label'].view(-1), num_classes=self.num_classes)
                               for x in outputs]).mean()

        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)

    def test_step(self,
                  batch: Dict = None,
                  batch_idx: int = None) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        labels = batch['label']

        return {'logits': logits, 'labels': labels}

    def test_epoch_end(
            self, outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
    ) -> None:
        logits = torch.cat([x['logits'].view(-1, self.num_classes) for x in outputs], dim=0)
        labels = torch.cat([x['labels'].view(-1) for x in outputs], dim=0)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc = accuracy(pred=preds,
                       target=labels,
                       num_classes=self.num_classes)
        pr = precision(pred=preds,
                       target=labels,
                       num_classes=self.num_classes,
                       class_reduction='none')
        rc = recall(pred=preds,
                    target=labels,
                    num_classes=self.num_classes,
                    class_reduction='none')
        f1 = f1_score(pred=preds,
                      target=labels,
                      num_classes=self.num_classes,
                      class_reduction='none')
        auc = auroc(pred=preds,
                    target=labels)

        print(f'Test loss: {loss:.4f}')
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {pr[1]:.4f}')
        print(f'Recall: {rc[1]:.4f}')
        print(f'F1: {f1[1]:.4f}')
        print(f'Auc: {auc:.4f}')