from typing import Dict
import json
from pathlib import Path

import jsonlines
import torch
from torch.nn import Module
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
from absl import app, flags, logging
from allennlp.data import PyTorchDataLoader, Vocabulary
from allennlp.modules import Embedding, FeedForward
from allennlp.modules.seq2vec_encoders import (
    BagOfEmbeddingsEncoder, LstmSeq2VecEncoder
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.nn import util as nn_util
from allennlp.training.metrics import F1Measure, Auc
from tqdm import tqdm
from more_itertools import chunked

from dataset_readers import TokenReader, JamoReader, TransformerReader
from models import TokenModel, JamoCnnModel, TokenTransformerModel
from modules import CnnDialogueEncoder, TransformerEmbeddings

FLAGS = flags.FLAGS

flags.DEFINE_string('model_state_path', default=None,
                    help='Path to the model state (weights) file')
flags.DEFINE_string('params_path', default=None,
                    help='Path to the params file')
flags.DEFINE_string('vocab_dir', default=None,
                    help='Path to the directory containing vocab files')
flags.DEFINE_string('model_dir', default=None,
                    help='Directory containing saved model files. '
                         'If this flag is given, '
                         'model_state_path, params_path, and vocab_dir '
                         'should not be set.')
flags.DEFINE_string('test_data_path', default=None,
                    help='Path to the test data')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save prediction results')
flags.DEFINE_integer('cuda_device', default=-1,
                     help='If given, uses this CUDA device in evaluation')
flags.DEFINE_integer('batch_size', default=64,
                     help='Batch size')


def create_sentence_encoder(params: Dict = None):
    if params['sentence_encoder_type'] == 'boe':
        encoder = BagOfEmbeddingsEncoder(
            embedding_dim=params['token_embedding_dim'],
            averaged=True
        )
    elif params['sentence_encoder_type'] == 'lstm':
        encoder = LstmSeq2VecEncoder(
            input_size=params['token_embedding_dim'],
            hidden_size=params['sentence_encoder_hidden_dim'],
            num_layers=1,
            bidirectional=False
        )
    elif params['sentence_encoder_type'] == 'bilstm':
        encoder = LstmSeq2VecEncoder(
            input_size=params['token_embedding_dim'],
            hidden_size=params['sentence_encoder_hidden_dim'],
            num_layers=1,
            bidirectional=True
        )
    else:
        raise ValueError('Unknown sentence_encoder_type')
    return encoder


def create_dialogue_encoder(params: Dict = None,
                            extra_dim: int = None):
    if params['sentence_encoder_type'] == 'bilstm':
        params['sentence_encoder_hidden_dim'] = params['sentence_encoder_hidden_dim'] * 2
    elif params['sentence_encoder_type'] == 'boe':
        params['sentence_encoder_hidden_dim'] = params['token_embedding_dim']
    if params['dialogue_encoder_type'] == 'boe':
        input_dim = params['sentence_encoder_hidden_dim'] + extra_dim
        encoder = BagOfEmbeddingsEncoder(
            embedding_dim=input_dim,
            averaged=True
        )
    elif params['dialogue_encoder_type'] == 'lstm':
        input_dim = params['sentence_encoder_hidden_dim'] + extra_dim
        encoder = LstmSeq2VecEncoder(
            input_size=input_dim,
            hidden_size=params['dialogue_encoder_hidden_dim'],
            num_layers=1,
            bidirectional=False
        )
    else:
        raise ValueError('Unknown dialogue_encoder_type')
    return encoder


def create_dialogue_cnn_encoder(params: Dict = None) -> Module:
    encoder = CnnDialogueEncoder(
        embedding_dim=params['token_embedding_dim'],
        num_filters=params['dialogue_cnn_encoder_num_filters'],
        ngram_filter_sizes=(3, 4, 5)
    )
    return encoder

def create_transformer_encoder_layer(params: Dict = None) -> Module:
    encoder = TransformerEncoderLayer(
        d_model=params['token_embedding_dim'],
        nhead=params['transformer_encoder_layer_nhead']
    )
    return encoder

def main(argv):
    model_state_path = FLAGS.model_state_path
    params_path = FLAGS.params_path
    vocab_dir = FLAGS.vocab_dir

    if not model_state_path:
        model_state_path = str(Path(FLAGS.model_dir) / 'best.th')
    if not params_path:
        params_path = str(Path(FLAGS.model_dir) / 'params.json')
    if not vocab_dir:
        vocab_dir = str(Path(FLAGS.model_dir) / 'vocab')

    logging.info(f'Model state path: {model_state_path}')
    logging.info(f'Params path: {params_path}')
    logging.info(f'Vocabulary directory: {vocab_dir}')

    with open(params_path, 'r') as f:
        params = json.load(f)

    logging.info('Loading vocabulary files')
    vocab = Vocabulary.from_files(vocab_dir)

    logging.info('Creating model')

    if params['model_type'] == 'token':
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders={
                'tokens': Embedding(embedding_dim=params['token_embedding_dim'],
                                    num_embeddings=vocab.get_vocab_size('tokens'))
            }
        )
        dialogue_extra_dim = 0
        dialogue_extra_dim += params['is_buyer_embedding_dim']

        sentence_encoder = create_sentence_encoder(params=params)
        dialogue_encoder = create_dialogue_encoder(params=params,
                                                   extra_dim=dialogue_extra_dim)
        discriminator = FeedForward(
            input_dim=dialogue_encoder.get_output_dim(),
            num_layers=2,
            hidden_dims=[params['discriminator_hidden_dim'],
                         vocab.get_vocab_size('labels')],
            activations=[Activation.by_name('relu')(),
                         Activation.by_name('linear')()]
        )
        model = TokenModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            sentence_encoder=sentence_encoder,
            dialogue_encoder=dialogue_encoder,
            discriminator=discriminator,
            is_buyer_embedding_dim=params['is_buyer_embedding_dim']
        )
    elif params['model_type'] == 'jamo_cnn':
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders={
                'tokens': Embedding(embedding_dim=params['token_embedding_dim'],
                                    num_embeddings=vocab.get_vocab_size('tokens'))
            }
        )
        dialogue_encoder = create_dialogue_cnn_encoder(params=params)
        discriminator = FeedForward(
            input_dim=dialogue_encoder.get_output_dim(),
            num_layers=2,
            hidden_dims=[params['discriminator_hidden_dim'],
                         vocab.get_vocab_size('labels')],
            activations=[Activation.by_name('relu')(),
                         Activation.by_name('linear')()]
        )
        model = JamoCnnModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            dialogue_encoder=dialogue_encoder,
            discriminator=discriminator
        )
    elif params['model_type'] == 'token_transformer':
        text_field_embedder = TransformerEmbeddings(
            vocab_size=vocab.get_vocab_size('tokens'),
            embedding_dim=params['token_embedding_dim']
        )
        transformer_encoder_layers = create_transformer_encoder_layer(params=params)
        dialogue_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layers,
                                              num_layers=params['transformer_encoder_nlayers'])
        discriminator = FeedForward(
            input_dim=params['token_embedding_dim'],
            num_layers=2,
            hidden_dims=[params['discriminator_hidden_dim'],
                         vocab.get_vocab_size('labels')],
            activations=[Activation.by_name('relu')(),
                         Activation.by_name('linear')()]
        )
        model = TokenTransformerModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            dialogue_encoder=dialogue_encoder,
            discriminator=discriminator
        )
    else:
        raise ValueError('Unknown model type')

    logging.info('Loading model states')
    model_state = torch.load(
        model_state_path, map_location=torch.device('cpu')
    )
    model.load_state_dict(model_state)

    model.eval()
    torch.set_grad_enabled(False)

    if FLAGS.cuda_device >= 0:
        model = model.to(FLAGS.cuda_device)

    logging.info(f'Reading data from {FLAGS.test_data_path}')
    test_json_data = []
    with jsonlines.open(FLAGS.test_data_path, 'r') as f:
        for obj in f:
            test_json_data.append(obj)

    if params['model_type'] == 'token':
        reader = TokenReader()
    elif params['model_type'] == 'jamo_cnn':
        reader = JamoReader(maximum_dialogue_length=params['max_dialogue_length'])
    elif params['model_type'] == 'token_transformer':
        reader = TransformerReader()
    else:
        raise ValueError('Unknown model type')
    dataset = reader.read(FLAGS.test_data_path)
    dataset.index_with(vocab)
    data_loader = PyTorchDataLoader(dataset=dataset, batch_size=FLAGS.batch_size)
    test_json_batches = list(chunked(test_json_data, FLAGS.batch_size))

    logging.info('Running prediction')
    logging.info(f'Will save results to {FLAGS.output_path}')
    output_file = jsonlines.open(FLAGS.output_path, 'w')
    for i, batch in enumerate(tqdm(data_loader)):
        batch = nn_util.move_to_device(batch, FLAGS.cuda_device)
        batch_output_dict = model(**batch)
        predicted_label_inds = (
            batch_output_dict['logits'].argmax(dim=-1).tolist()
        )
        test_json_objs = test_json_batches[i]
        assert len(predicted_label_inds) == len(test_json_objs)
        for obj, label_ind in zip(test_json_objs, predicted_label_inds):
            label = vocab.get_token_from_index(label_ind, namespace='labels')
            obj['predicted_label'] = label
        output_file.write_all(test_json_objs)
    output_file.close()


def validate_model_path_flags(flags_dict):
    if flags_dict['model_dir'] is None:
        return not (flags_dict['model_state_path'] is None
                    or flags_dict['params_path'] is None
                    or flags_dict['vocab_dir'] is None)
    return True


if __name__ == '__main__':
    flags.mark_flags_as_required(['test_data_path'])
    flags.register_multi_flags_validator(
        flag_names=['model_state_path', 'params_path', 'vocab_dir',
                    'model_dir'],
        multi_flags_checker=validate_model_path_flags,
        message='If model_dir is not given, model_state_path, params_path, '
                'and vocab_dir should be provided.'
    )
    app.run(main)
