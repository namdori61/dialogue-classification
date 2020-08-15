"""
Subsamples the entire data to make the numbers of fraud and non-fraud
data identical.
"""
import random

import jsonlines
from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to input data')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save output data')

random.seed(123)


def main(argv):
    # Count number of data
    num_data = 0
    fraud_indices_set = set()
    with jsonlines.open(FLAGS.input_path, 'r') as f:
        for line_num, obj in tqdm(enumerate(f), desc='Counting data'):
            num_data += 1
            if obj['label'] == 1:
                fraud_indices_set.add(line_num)

    logging.info(f'Number of data: {num_data}')
    logging.info(f'Number of fraud data: {len(fraud_indices_set)}')

    logging.info(f'Will sample {len(fraud_indices_set)} non-fraud data')

    non_fraud_indices_set = set(range(num_data)) - fraud_indices_set
    non_fraud_indices_list = list(non_fraud_indices_set)
    random.shuffle(non_fraud_indices_list)
    sampled_non_fraud_indices_list = (
        non_fraud_indices_list[:len(fraud_indices_set)]
    )

    survived_indices_set = (
        set.union(set(sampled_non_fraud_indices_list), fraud_indices_set)
    )

    input_file = open(FLAGS.input_path, 'r')
    output_file = open(FLAGS.output_path, 'w')

    for line_num, line in tqdm(enumerate(input_file), desc='Sampling'):
        if line_num in survived_indices_set:
            output_file.write(line)


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'output_path'])
    app.run(main)
