"""
Tokenize a sentence into tokens using supervised (external) tokenizer.

Input: JSON Lines file obtained from split_into_sentences
Output: JSON Lines file where each sentence is tokenized
"""
import jsonlines
from absl import app, flags
from konlpy import tag
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to input file')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save output file')
flags.DEFINE_enum('method', default=None,
                  enum_values=['komoran', 'okt', 'mecab', 'hnn', 'kkma'],
                  help='Tokenization method')


def main(argv):
    if FLAGS.method == 'komoran':
        tokenizer = tag.Komoran()
    elif FLAGS.method == 'okt':
        tokenizer = tag.Okt()
    elif FLAGS.method == 'mecab':
        tokenizer = tag.Mecab()
    elif FLAGS.method == 'hnn':
        tokenizer = tag.Hannanum()
    elif FLAGS.method == 'kkma':
        tokenizer = tag.Kkma()
    else:
        raise ValueError('Unknown method')

    input_reader = jsonlines.open(FLAGS.input_path, 'r')
    output_writer = jsonlines.open(FLAGS.output_path, 'w')

    for obj in tqdm(input_reader, desc='Tokenizing'):
        messages = obj['messages'].copy()
        for message in messages:
            text = message['text']
            tokens = tokenizer.morphs(text)
            token_objs = [{'token': token} for token in tokens]
            message['text'] = token_objs
        output_writer.write(obj)


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'output_path', 'method'])
    app.run(main)
