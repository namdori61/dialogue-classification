"""
Tokenize a sentence into tokens using unsupervised tokenizer
trained using a given corpus.

Input: JSON Lines file obtained from split_into_sentences
Output: JSON Lines file where each sentence is tokenized
"""
import math
from pathlib import Path

import jsonlines
from absl import app, flags, logging
from tqdm import tqdm

import sentencepiece as spm
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soyspacing.countbase import CountSpace

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to input file')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save output file')
flags.DEFINE_enum('mode', default=None,
                  enum_values=['train', 'tokenize'],
                  help='Set this to "train" when training a tokenizer, '
                       'and to "tokenize" to tokenize text using '
                       'a trained tokenizer')
flags.DEFINE_enum('method', default=None,
                  enum_values=['soynlp', 'bpe'],
                  help='Unsupervised tokenization method')
flags.DEFINE_string('model_save_dir', default=None,
                    help='Path to save trained tokenizer model files')
flags.DEFINE_string('model_load_dir', default=None,
                    help='Path to load trained tokenizer model files')
flags.DEFINE_boolean('correct_spacing', default=False,
                     help='Whether to correct spacing using soyspacing')
flags.DEFINE_integer('bpe_vocab_size', default=None,
                     help='Output vocabulary size of BPE tokenizer')


def make_sentence_corpus(input_path, output_path):
    input_reader = jsonlines.open(input_path, 'r')
    with open(output_path, 'w') as f:
        for input_obj in tqdm(input_reader, desc='Processing corpus'):
            messages = input_obj['messages']
            for message in messages:
                text = message['text']
                f.write(text + '\n')


def train_soyspacing(input_path, model_path):
    model = CountSpace()
    model.train(input_path)
    model.save_model(model_path, json_format=False)


def apply_soyspacing(input_path, model_path, output_path):
    model = CountSpace()
    model.load_model(model_path, json_format=False)
    input_file = open(input_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'w', encoding='utf-8')
    for sentence in input_file:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_corrected, _ = model.correct(sentence)
        output_file.write(sent_corrected)
        output_file.write('\n')


def train_tokenizer(input_path, method, save_dir, bpe_vocab_size=None):
    if method == 'soynlp':
        with open(input_path, 'r') as f:
            sentences = f.readlines()
        sentences = [sent.strip() for sent in sentences]
        word_extractor = WordExtractor(min_frequency=100,
                                       min_cohesion_forward=0.05,
                                       min_right_branching_entropy=0.0)
        word_extractor.train(sentences)
        model_path = str(Path(save_dir) / 'tokenizer.model')
        word_extractor.save(model_path)
    elif FLAGS.method == 'bpe':
        model_prefix = Path(save_dir) / 'tokenizer'
        spm.SentencePieceTrainer.train(input=input_path,
                                       model_prefix=model_prefix,
                                       model_type='bpe',
                                       vocab_size=bpe_vocab_size,
                                       character_coverage=0.9995)
    else:
        raise ValueError('Unknown tokenize method')


def load_tokenizer(method, model_path):
    if method == 'soynlp':
        word_extractor = WordExtractor(min_frequency=100,
                                       min_cohesion_forward=0.05,
                                       min_right_branching_entropy=0.0)
        word_extractor.load(model_path)
        scores = word_extractor.word_scores()
        scores = {key: (scores[key].cohesion_forward
                        * math.exp(scores[key].right_branching_entropy))
                  for key in scores.keys()}
        tokenizer = MaxScoreTokenizer(scores=scores)
    elif method == 'bpe':
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(model_path)
    else:
        raise ValueError('Unknown tokenize method')
    return tokenizer


def validate_flags_model_dir(flags_dict):
    if flags_dict['mode'] == 'train':
        if flags_dict['model_save_dir'] is None:
            return False
    else:
        if flags_dict['model_load_dir'] is None:
            return False
    return True


def validate_flags_output_path(flags_dict):
    if flags_dict['mode'] == 'tokenize':
        if flags_dict['output_path'] is None:
            return False
    return True


def validate_flags_bpe_vocab_size(flags_dict):
    if flags_dict['mode'] == 'train':
        if flags_dict['method'] == 'bpe':
            if flags_dict['bpe_vocab_size'] is None:
                return False
    return True


def main(argv):
    if FLAGS.mode == 'train':
        logging.info('Will be run as training mode')

        model_save_dir = Path(FLAGS.model_save_dir)
        model_save_dir.mkdir(parents=True)
        logging.info(f'Making sentence corpus...')
        sent_corpus_path = str(model_save_dir / 'sent_corpus.txt')
        make_sentence_corpus(input_path=FLAGS.input_path,
                             output_path=sent_corpus_path)
        logging.info(f'Saved sentence corpus to: {sent_corpus_path}')

        train_corpus_path = sent_corpus_path

        if FLAGS.correct_spacing:
            logging.info('Training soyspacing...')
            soyspacing_model_path = str(model_save_dir / 'soyspacing.model')
            spaced_sent_corpus_path = (
                model_save_dir / 'spaced_sent_corpus.txt'
            )
            train_soyspacing(input_path=sent_corpus_path,
                             model_path=soyspacing_model_path)
            apply_soyspacing(input_path=sent_corpus_path,
                             model_path=soyspacing_model_path,
                             output_path=spaced_sent_corpus_path)
            train_corpus_path = spaced_sent_corpus_path
            logging.info(f'Saved spaced sentence corpus to: '
                         f'{spaced_sent_corpus_path}')

        logging.info(f'Training {FLAGS.method} tokenizer '
                     f'using corpus {train_corpus_path}...')
        train_tokenizer(input_path=train_corpus_path,
                        method=FLAGS.method,
                        save_dir=FLAGS.model_save_dir,
                        bpe_vocab_size=FLAGS.bpe_vocab_size)
        logging.info(f'Training done! Saved model files to {model_save_dir}')
    else:
        logging.info('Will be run as tokenization mode')

        tokenizer_model_path = (
            str(Path(FLAGS.model_load_dir) / 'tokenizer.model')
        )
        logging.info(f'Loading tokenizer model from {tokenizer_model_path}')
        tokenizer = load_tokenizer(method=FLAGS.method,
                                   model_path=tokenizer_model_path)

        input_reader = jsonlines.open(FLAGS.input_path, 'r')
        output_writer = jsonlines.open(FLAGS.output_path, 'w')

        if FLAGS.method == 'soynlp':
            tokenize_fn = tokenizer.tokenize
        elif FLAGS.method == 'bpe':
            tokenize_fn = tokenizer.encode_as_pieces
        else:
            raise ValueError('Unknown tokenize method')

        for obj in tqdm(input_reader, desc='Tokenizing'):
            messages = obj['messages']
            for message in messages:
                text = message['text']
                tokens = tokenize_fn(text)
                token_objs = [{'token': token} for token in tokens]
                message['text'] = token_objs
            output_writer.write(obj)

        input_reader.close()
        output_writer.close()


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'method', 'mode'])

    flags.register_multi_flags_validator(
        flag_names=['mode', 'model_save_dir', 'model_load_dir'],
        multi_flags_checker=validate_flags_model_dir,
        message='When mode is "train", model_save_dir should be provided, '
                'and model_load_dir should be provided '
                'when model is "tokenize".')
    flags.register_multi_flags_validator(
        flag_names=['mode', 'output_path'],
        multi_flags_checker=validate_flags_output_path,
        message='In "tokenize" mode, output_path should be given.')
    flags.register_multi_flags_validator(
        flag_names=['mode', 'method', 'bpe_vocab_size'],
        multi_flags_checker=validate_flags_bpe_vocab_size,
        message='When mode is "train" and "bpe" is used as method, '
                'bpe_vocab_size must be given.')

    app.run(main)
