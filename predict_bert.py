from absl import app, flags, logging

from tqdm import tqdm
import jsonlines
from more_itertools import chunked
import torch
from transformers import BertTokenizer
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import DataLoader, SequentialSampler
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from dataset_readers import BertReader, KoBertReader
from models import TokenBertModel, TokenKoBertModel


FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', default=None,
                    help='Model to evaluate (BERT, KoBERT, KcBERT)')
flags.DEFINE_string('model_state_path', default=None,
                    help='Path to the model state (weights) file')
flags.DEFINE_string('hparams_path', default=None,
                    help='Path to the model hparams file')
flags.DEFINE_string('test_data_path', default=None,
                    help='Path to the test data')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save prediction results')
flags.DEFINE_integer('maximum_length', default=512,
                     help='If given, uses this maximum length to data loading')
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
                             tokenizer=tokenizer,
                             maximum_length=FLAGS.maximum_length)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=FLAGS.batch_size,
                                num_workers=FLAGS.num_workers)
    elif FLAGS.model_type == 'KoBERT':
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        model = TokenKoBertModel.load_from_checkpoint(checkpoint_path=FLAGS.model_state_path,
                                                      hparams_file=FLAGS.hparams_path)
        dataset = KoBertReader(file_path=FLAGS.test_data_path,
                               tokenizer=tokenizer,
                               maximum_length=FLAGS.maximum_length)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=FLAGS.batch_size,
                                num_workers=FLAGS.num_workers)
    elif FLAGS.model_type == 'KcBERT':
        tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
        model = TokenBertModel.load_from_checkpoint(checkpoint_path=FLAGS.model_state_path,
                                                    hparams_file=FLAGS.hparams_path)
        dataset = BertReader(file_path=FLAGS.test_data_path,
                             tokenizer=tokenizer,
                             maximum_length=FLAGS.maximum_length)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=FLAGS.batch_size,
                                num_workers=FLAGS.num_workers)
    else:
        raise ValueError('Unknown model type')

    logging.info(f'Reading data from {FLAGS.test_data_path}')
    test_json_data = []
    with jsonlines.open(FLAGS.test_data_path, 'r') as f:
        for obj in f:
            test_json_data.append(obj)

    test_json_batches = list(chunked(test_json_data, FLAGS.batch_size))

    model.freeze()
    if FLAGS.cuda_device > 0:
        model.to('cuda')
        #model.forward = auto_move_data(model.forward)

    logging.info('Running prediction')
    logging.info(f'Will save results to {FLAGS.output_path}')
    output_file = jsonlines.open(FLAGS.output_path, 'w')
    for i, batch in enumerate(tqdm(dataloader)):
        if FLAGS.cuda_device > 0:
            batch = {k : v.to('cuda') for k, v in batch.items()}
        logits = model.forward(batch)
        predicted_label_inds = logits.argmax(dim=-1).tolist()

        test_json_objs = test_json_batches[i]

        assert len(predicted_label_inds) == len(test_json_objs)

        for obj, label_ind in zip(test_json_objs, predicted_label_inds):
            obj['predicted_label'] = label_ind
        output_file.write_all(test_json_objs)
    output_file.close()


if __name__ == '__main__':
    flags.mark_flags_as_required(['model_type', 'model_state_path', 'test_data_path'])
    app.run(main)
