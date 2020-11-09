from absl import app, flags, logging

from transformers import BertTokenizer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, SequentialSampler
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from dataset_readers import BertReader, KoBertReader
from models import TokenBertModel, TokenKoBertModel


FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', default=None,
                    help='Model to evaluate (BERT, KoBERT)')
flags.DEFINE_string('model_state_path', default=None,
                    help='Path to the model state (weights) file')
flags.DEFINE_string('hparams_path', default=None,
                    help='Path to the model hparams file')
flags.DEFINE_string('test_data_path', default=None,
                    help='Path to the test data')
flags.DEFINE_integer('cuda_device', default=-1,
                     help='If given, uses this CUDA device in evaluation')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_integer('batch_size', default=64,
                     help='Batch size')


def main(argv):
    logging.info('Creating model')
    logging.info(f'Loading model states at {FLAGS.model_state_path}')
    logging.info(f'Loading model hparams at {FLAGS.hparams_path}')
    if FLAGS.model_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = TokenBertModel.load_from_checkpoint(checkpoint_path=FLAGS.model_state_path,
                                                    hparams_file=FLAGS.hparams_path)
        dataset = BertReader(file_path=FLAGS.test_data_path,
                             tokenizer=tokenizer)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=FLAGS.batch_size,
                                num_workers=FLAGS.num_workers)
    elif FLAGS.model == 'KoBERT':
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        model = TokenKoBertModel.load_from_checkpoint(checkpoint_path=FLAGS.model_state_path,
                                                      hparams_file=FLAGS.hparams_path)
        dataset = KoBertReader(file_path=FLAGS.test_data_path,
                               tokenizer=tokenizer)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=FLAGS.batch_size,
                                num_workers=FLAGS.num_workers)
    else:
        raise ValueError('Unknown model type')

    logging.info('Running evaluation')
    if FLAGS.cuda_device > 0:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device)
    else:
        trainer = Trainer(deterministic=True)

    trainer.test(model=model,
                 test_dataloaders=dataloader)


if __name__ == '__main__':
    flags.mark_flags_as_required(['model_type', 'model_state_path', 'hparams_path', 'test_data_path'])
    app.run(main)
