import re

import jsonlines
from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to input jsonl file')
flags.DEFINE_string('output_path', default=None, help='Path to save output')


def transform_char_into_jamo(char):
    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ',
                    'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ',
                     'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ',
                     'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ',
                     'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    jamo_list = []
    if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
        if ord(char) >= BASE_CODE:
            char_code = ord(char) - BASE_CODE
            cho_idx = int(char_code / CHOSUNG)
            jamo_list.append(CHOSUNG_LIST[cho_idx])

            jung_idx = int((char_code - (CHOSUNG * cho_idx)) / JUNGSUNG)
            jamo_list.append(JUNGSUNG_LIST[jung_idx])

            jong_idx = int((char_code - (CHOSUNG * cho_idx) -
                            (JUNGSUNG * jung_idx)))
            if jong_idx == 0:
                jamo_list.append('#')
            else:
                jamo_list.append(JONGSUNG_LIST[jong_idx])
        else:
            jamo_list.append(char)
    else:
        jamo_list.append(char)

    return jamo_list


def tranform_token_into_jamo(token):
    jamo_list = []
    for char in list(token):
        jamo_list.extend(transform_char_into_jamo(char))
    return jamo_list


def transform_text_into_jamo(text):
    jamo_list = []
    for char in list(text.replace(' ', '')):
        jamo_list.extend(transform_char_into_jamo(char))
    jamo_list = [{'token': token} for token in jamo_list]
    return jamo_list


def main(argv):
    input_path = FLAGS.input_path
    output_path = FLAGS.output_path

    input_reader = jsonlines.open(input_path, 'r')
    output_writer = jsonlines.open(output_path, 'w')
    num_lines = 0
    for input_obj in tqdm(input_reader, desc='Processing'):
        output_obj = input_obj.copy()
        messages = output_obj['messages']
        for message in messages:
            text = message['text']
            message['text'] = transform_text_into_jamo(text)
        output_writer.write(output_obj)
        num_lines += 1
    input_reader.close()
    output_writer.close()
    logging.info(f'Processed {num_lines} lines')


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'output_path'])
    app.run(main)
