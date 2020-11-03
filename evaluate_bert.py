from absl import app, flags, logging

from transformers import BertTokenizer
from pytorch_lightning import Trainer

from models import TokenBertModel


FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', default=None,
                    help='Model to evaluate (BERT, KoBERT)')
flags.DEFINE_string('model_state_path', default=None,
                    help='Path to the model state (weights) file')
flags.DEFINE_string('test_data_path', default=None,
                    help='Path to the test data')
flags.DEFINE_integer('cuda_device', default=-1,
                     help='If given, uses this CUDA device in evaluation')
flags.DEFINE_integer('batch_size', default=64,
                     help='Batch size')


def main(argv):
    logging.info('Creating model')
    if FLAGS.model_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = TokenBertModel(model='bert-base-multilingual-cased',
                               tokenizer=tokenizer,
                               test_path=FLAGS.test_data_path,
                               batch_size=FLAGS.batch_size)
    else:
        raise ValueError('Unknown model type')

    logging.info(f'Loading model states at {FLAGS.model_state_path}')
    model.load_from_checkpoint(FLAGS.model_state_path)

    logging.info('Running evaluation')
    if FLAGS.cuda_device > 0:
        trainer = Trainer(gpus=FLAGS.cuda_device)
    else:
        trainer = Trainer()

    trainer.test(model=model)


if __name__ == '__main__':
    flags.mark_flags_as_required(['model_type', 'model_state_path', 'test_data_path'])
    app.run(main)
