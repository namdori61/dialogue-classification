from typing import Union
from configparser import ConfigParser

from absl import app, flags, logging
import torch
from transformers import BertTokenizer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from knockknock import telegram_sender

from models import TokenBertModel


FLAGS = flags.FLAGS

flags.DEFINE_string('train_data_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_string('dev_data_path', default=None,
                    help='Path to the development dataset')
flags.DEFINE_string('test_data_path', default=None,
                    help='Path to the test dataset')
flags.DEFINE_string('model', default=None,
                    help='Model to train (BERT, KoBERT)')
flags.DEFINE_string('save_dir', default=None,
                    help='Path to save model')
flags.DEFINE_string('version', default=None,
                    help='Explain experiment version')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_integer('max_epochs', default=10,
                     help='If given, uses this max epochs in training')
flags.DEFINE_integer('batch_size', default=4,
                     help='If given, uses this batch size in training')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_float('lr', default=2e-5,
                   help='If given, uses this learning rate in training')
flags.DEFINE_float('weight_decay', default=0.1,
                   help='If given, uses this weight decay in training')
flags.DEFINE_integer('warm_up', default=500,
                     help='If given, uses this warm up in training')
flags.DEFINE_string('config_path', default=None,
                    help='Path to the config file')


def main(argv):
    if FLAGS.model == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = TokenBertModel(model='bert-base-multilingual-cased',
                               tokenizer=tokenizer,
                               train_path=FLAGS.train_data_path,
                               dev_path=FLAGS.dev_data_path,
                               test_path=FLAGS.test_data_path,
                               batch_size=FLAGS.batch_size,
                               num_workers=FLAGS.num_workers,
                               lr=FLAGS.lr,
                               weight_decay=FLAGS.weight_decay,
                               warm_up=FLAGS.warm_up)
    else:
        raise ValueError('Unknown model type')

    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        filepath=FLAGS.save_dir,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        strict=False,
        verbose=False,
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=FLAGS.save_dir,
        name='logs_' + FLAGS.model,
        version=FLAGS.version
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if FLAGS.config_path is not None:
        parser = ConfigParser()
        parser.read(FLAGS.config_path)

    @telegram_sender(token=parser.get('telegram', 'token'),
                     chat_id=parser.get('telegram', 'chat_id'))
    def train_notify(trainer: Trainer = None,
                     model: Union[LightningModule] = None) -> None:
        trainer.fit(model)

    if FLAGS.cuda_device > 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          distributed_backend='ddp',
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_monitor])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    elif FLAGS.cuda_device == 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_monitor])
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    else:
        trainer = Trainer(deterministic=True,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop,
                          max_epochs=FLAGS.max_epochs,
                          logger=logger,
                          callbacks=[lr_monitor])
        logging.info('No GPU available, using the CPU instead.')
    if FLAGS.config_path is not None:
        train_notify(trainer=trainer,
                     model=model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'train_data_path', 'model', 'save_dir', 'version'
    ])
    app.run(main)