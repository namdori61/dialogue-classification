"""
Divides the dataset_readers into training, development, and test split.
"""
import random
from pathlib import Path

from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to the input data')
flags.DEFINE_string('save_dir', default=None,
                    help='Directory to save the splits')
flags.DEFINE_integer('num_dev_data', default=10000,
                     help='The size of the development set')
flags.DEFINE_integer('num_test_data', default=10000,
                     help='The size of the test set')

random.seed(1234)


def main(argv):
    # Count number of data
    num_data = 0
    with open(FLAGS.input_path, 'r') as f:
        for line in tqdm(f, desc='Counting data'):
            num_data += 1

    num_dev_data = FLAGS.num_dev_data
    num_test_data = FLAGS.num_test_data
    num_train_data = num_data - num_dev_data - num_test_data
    logging.info(f'# training samples: {num_train_data}')
    logging.info(f'# development samples: {num_dev_data}')
    logging.info(f'# test samples: {num_test_data}')

    indices = list(range(num_data))
    random.shuffle(indices)

    train_indices = set(indices[:num_train_data])
    dev_indices = set(indices[num_train_data:-num_test_data])
    test_indices = set(indices[-num_test_data:])

    assert len(train_indices) == num_train_data
    assert len(dev_indices) == num_dev_data
    assert len(test_indices) == num_test_data

    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir()
    train_file = open(save_dir / 'train.jsonl', 'w')
    dev_file = open(save_dir / 'dev.jsonl', 'w')
    test_file = open(save_dir / 'test.jsonl', 'w')

    with open(FLAGS.input_path, 'r') as f:
        for line_num, line in tqdm(enumerate(f), desc='Splitting dataset_readers'):
            if line_num in train_indices:
                file_to_write = train_file
            elif line_num in dev_indices:
                file_to_write = dev_file
            else:
                file_to_write = test_file
            file_to_write.write(line)


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'save_dir'])
    app.run(main)
