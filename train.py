"""
Trains a model.
"""
import json
from pathlib import Path

import optuna
from absl import app, flags, logging
from allennlp.data import PyTorchDataLoader, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import (
    BagOfEmbeddingsEncoder, LstmSeq2VecEncoder, Seq2VecEncoder
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.training import Checkpointer, GradientDescentTrainer
from torch.optim import Adam

from dataset_readers import TokenReader
from models import TokenModel

FLAGS = flags.FLAGS

flags.DEFINE_enum('model_type', default=None,
                  enum_values=['token'],
                  help='Model type')
flags.DEFINE_enum('sentence_encoder_type', default=None,
                  enum_values=['boe', 'lstm', 'bilstm'],
                  help='Sentence encoder type')
flags.DEFINE_enum('dialogue_encoder_type', default=None,
                  enum_values=['boe', 'lstm'],
                  help='Dialogue encoder type')
flags.DEFINE_integer('min_token_count', default=None,
                     help='Only use tokens appearing in the dataset '
                          'more than this time for building vocabulary')
flags.DEFINE_string('train_data_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_string('dev_data_path', default=None,
                    help='Path to the development dataset')
flags.DEFINE_string('save_root_dir', default=None,
                    help='Root path to save  results')
flags.DEFINE_integer('batch_size', default=64, help='Batch size')
flags.DEFINE_integer('max_dialogue_length', default=None,
                     help='Maximum number of sentences per dialogue')
flags.DEFINE_integer('max_sentence_length', default=None,
                     help='Maximum number of words per sentence')
flags.DEFINE_string('optuna_study_name', default=None,
                    help='Name of Optuna study')
flags.DEFINE_integer('optuna_num_trials', default=100,
                     help='Number of trials to use in HPO')
flags.DEFINE_string('optuna_storage', default='sqlite:///optuna_studies.db',
                    help='Path to save Optuna results')
flags.DEFINE_integer('cuda_device', default=-1,
                     help='If given, uses this CUDA device in training')


def create_sentence_encoder(trial: optuna.Trial,
                            token_embedding_dim: int) -> Seq2VecEncoder:
    if FLAGS.sentence_encoder_type == 'boe':
        encoder = BagOfEmbeddingsEncoder(
            embedding_dim=token_embedding_dim,
            averaged=True
        )
    elif FLAGS.sentence_encoder_type == 'lstm':
        encoder = LstmSeq2VecEncoder(
            input_size=token_embedding_dim,
            hidden_size=trial.suggest_categorical(
                name='sentence_encoder_hidden_dim', choices=[256, 512]),
            num_layers=1,
            bidirectional=False
        )
    elif FLAGS.sentence_encoder_type == 'bilstm':
        encoder = LstmSeq2VecEncoder(
            input_size=token_embedding_dim,
            hidden_size=trial.suggest_categorical(
                name='sentence_encoder_hidden_dim', choices=[256, 512]),
            num_layers=1,
            bidirectional=True
        )
    else:
        raise ValueError('Unknown sentence_encoder_type')
    return encoder


def create_dialogue_encoder(trial: optuna.Trial,
                            sentence_encoder_output_dim: int,
                            extra_dim: int = 0
                            ) -> Seq2VecEncoder:
    if FLAGS.dialogue_encoder_type == 'boe':
        encoder = BagOfEmbeddingsEncoder(
            embedding_dim=sentence_encoder_output_dim + extra_dim,
            averaged=True
        )
    elif FLAGS.dialogue_encoder_type == 'lstm':
        input_dim = sentence_encoder_output_dim + extra_dim
        encoder = LstmSeq2VecEncoder(
            input_size=input_dim,
            hidden_size=trial.suggest_categorical(
                name='dialogue_encoder_hidden_dim', choices=[256, 512]
            ),
            num_layers=1,
            bidirectional=False
        )
    else:
        raise ValueError('Unknown dialogue_encoder_type')
    return encoder


def create_model(trial: optuna.Trial,
                 vocab: Vocabulary) -> Model:
    token_embedding_dim = trial.suggest_categorical(
        name='token_embedding_dim', choices=[150, 300]
    )

    dialogue_extra_dim = 0
    is_buyer_embedding_dim = trial.suggest_categorical(
        name='is_buyer_embedding_dim', choices=[32, 64]
    )
    dialogue_extra_dim += is_buyer_embedding_dim

    token_vocab_size = vocab.get_vocab_size('tokens')
    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders={
            'tokens': Embedding(embedding_dim=token_embedding_dim,
                                num_embeddings=token_vocab_size)
        }
    )

    sentence_encoder = create_sentence_encoder(
        trial=trial, token_embedding_dim=token_embedding_dim
    )

    dialogue_encoder = create_dialogue_encoder(
        trial=trial,
        sentence_encoder_output_dim=sentence_encoder.get_output_dim(),
        extra_dim=dialogue_extra_dim
    )

    discriminator = FeedForward(
        input_dim=dialogue_encoder.get_output_dim(),
        num_layers=2,
        hidden_dims=[trial.suggest_categorical(name='discriminator_hidden_dim',
                                               choices=[256, 512]),
                     vocab.get_vocab_size('labels')],
        activations=[Activation.by_name('relu')(),
                     Activation.by_name('linear')()],
        dropout=[trial.suggest_float(name='discriminator_dropout',
                                     low=0.0, high=0.5),
                 0.0]
    )

    embedding_dropout = trial.suggest_float(name='embedding_dropout',
                                            low=0.0, high=0.5)
    sentence_encoder_output_dropout = trial.suggest_float(
        name='sentence_encoder_output_dropout', low=0.0, high=0.5
    )
    dialogue_encoder_output_dropout = trial.suggest_float(
        name='dialogue_encoder_output_dropout', low=0.0, high=0.5
    )

    if FLAGS.model_type == 'token':
        model = TokenModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            sentence_encoder=sentence_encoder,
            dialogue_encoder=dialogue_encoder,
            discriminator=discriminator,
            is_buyer_embedding_dim=is_buyer_embedding_dim,
            embedding_dropout=embedding_dropout,
            sentence_encoder_output_dropout=sentence_encoder_output_dropout,
            dialogue_encoder_output_dropout=dialogue_encoder_output_dropout
        )
    else:
        raise ValueError('Unknown model_type')

    return model


def optimize(trial: optuna.Trial) -> float:
    reader = TokenReader(maximum_dialogue_length=FLAGS.max_dialogue_length,
                         maximum_sentence_length=FLAGS.max_sentence_length)

    train_dataset = reader.read(FLAGS.train_data_path)
    dev_dataset = reader.read(FLAGS.dev_data_path)

    train_loader = PyTorchDataLoader(dataset=train_dataset,
                                     batch_size=FLAGS.batch_size,
                                     shuffle=True)
    dev_loader = PyTorchDataLoader(dataset=dev_dataset,
                                   batch_size=FLAGS.batch_size,
                                   shuffle=False)

    vocab = Vocabulary.from_instances(
        train_dataset.instances, min_count={'tokens': FLAGS.min_token_count},
    )
    logging.info(f'Vocabulary size: {vocab.get_vocab_size()}')

    train_dataset.index_with(vocab)
    dev_dataset.index_with(vocab)

    trial_name = f'trial-{trial.number:02d}'
    save_dir = Path(FLAGS.save_root_dir) / trial_name
    save_dir.mkdir(parents=True)

    model = create_model(trial=trial, vocab=vocab)
    if FLAGS.cuda_device >= 0:
        model = model.to(FLAGS.cuda_device)

    optimizer = Adam(model.parameters())
    checkpointer = Checkpointer(serialization_dir=save_dir,
                                num_serialized_models_to_keep=1)

    params_path = save_dir / 'params.json'
    with open(params_path, 'w') as f:
        params = trial.params.copy()
        params['model_type'] = FLAGS.model_type
        params['sentence_encoder_type'] = FLAGS.sentence_encoder_type
        params['dialogue_encoder_type'] = FLAGS.dialogue_encoder_type
        params['max_dialogue_length'] = FLAGS.max_dialogue_length
        params['max_sentence_length'] = FLAGS.max_sentence_length
        params['min_token_count'] = FLAGS.min_token_count
        json.dump(params, f, sort_keys=True, indent=4)
        logging.info(f'Params: {params}')
        logging.info(f'Saved params file to {params_path}')
    vocab_dir = save_dir / 'vocab'
    vocab.save_to_files(save_dir / 'vocab')
    logging.info(f'Saved vocab files to {vocab_dir}')

    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        patience=2,
        validation_metric='+accuracy',
        validation_data_loader=dev_loader,
        num_epochs=20,
        serialization_dir=save_dir,
        checkpointer=checkpointer,
        cuda_device=FLAGS.cuda_device,
        grad_norm=5.0
    )
    metrics = trainer.train()
    return metrics['best_validation_accuracy']


def main(argv):
    study = optuna.create_study(storage=FLAGS.optuna_storage,
                                study_name=FLAGS.optuna_study_name,
                                direction='maximize',
                                load_if_exists=True)
    study.optimize(optimize, n_trials=FLAGS.optuna_num_trials)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'model_type', 'sentence_encoder_type', 'dialogue_encoder_type',
        'min_token_count',
        'train_data_path', 'dev_data_path', 'save_root_dir',
        'optuna_study_name'
    ])
    app.run(main)
