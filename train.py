"""
Trains a model.
"""
import json
from pathlib import Path
import pickle
from configparser import ConfigParser

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
from allennlp.training import Checkpointer, GradientDescentTrainer, TensorboardWriter
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from torch.nn import Module
from torch.optim import Adam
from torch.nn import TransformerEncoderLayer, TransformerEncoder


from dataset_readers import TokenReader, JamoReader, TransformerReader
from models import TokenModel, JamoCnnModel, TokenTransformerModel
from modules import CnnDialogueEncoder, TransformerEmbeddings

from knockknock import telegram_sender


FLAGS = flags.FLAGS

flags.DEFINE_enum('model_type', default=None,
                  enum_values=['token', 'jamo_cnn', 'token_transformer'],
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
flags.DEFINE_string('val_metric', default='accuracy',
                    help='Validation metric to optimize')
flags.DEFINE_string('optuna_study_name', default=None,
                    help='Name of Optuna study')
flags.DEFINE_integer('optuna_num_trials', default=100,
                     help='Number of trials to use in HPO')
flags.DEFINE_string('optuna_storage', default='sqlite:///optuna_studies.db',
                    help='Path to save Optuna results')
flags.DEFINE_integer('cuda_device', default=-1,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_float('lr', default=1e-3,
                   help='Learning rate used in training')
flags.DEFINE_integer('warmup_steps', default=5000,
                     help='The number of warm up steps used to lr scheduler')
flags.DEFINE_integer('num_epochs', default=20,
                     help='The number of epochs used in training')
flags.DEFINE_string('config_path', default=None,
                    help='Path to the config file')
flags.DEFINE_integer('max_epochs', default=20,
                     help='If given, uses this max epochs in training')


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

def create_dialogue_cnn_encoder(trial: optuna.Trial,
                                token_embedding_dim: int) -> Module:
    encoder = CnnDialogueEncoder(
        embedding_dim=token_embedding_dim,
        num_filters=trial.suggest_categorical(
            name='dialogue_cnn_encoder_num_filters', choices=[50, 100]
        ),
        ngram_filter_sizes=(3, 4, 5)
    )
    return encoder

def create_transformer_encoder_layer(trial: optuna.Trial,
                                     token_embedding_dim: int) -> Module:
    encoder = TransformerEncoderLayer(
        d_model=token_embedding_dim,
        nhead=trial.suggest_categorical(
            name='transformer_encoder_layer_nhead', choices=[4, 8]
        ),
        dropout=trial.suggest_float(
            name='transformer_encoder_layer_dropout', low=0.0, high=0.5
        )
    )
    return encoder

def create_token_model(trial: optuna.Trial,
                       vocab: Vocabulary) -> Model:
    token_embedding_dim = trial.suggest_categorical(
        name='token_embedding_dim', choices=[150, 300]
    )
    token_vocab_size = vocab.get_vocab_size('tokens')
    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders={
            'tokens': Embedding(embedding_dim=token_embedding_dim,
                                num_embeddings=token_vocab_size)
        }
    )

    dialogue_extra_dim = 0
    is_buyer_embedding_dim = trial.suggest_categorical(
        name='is_buyer_embedding_dim', choices=[32, 64]
    )
    dialogue_extra_dim += is_buyer_embedding_dim

    sentence_encoder = create_sentence_encoder(
        trial=trial, token_embedding_dim=token_embedding_dim
    )

    dialogue_encoder = create_dialogue_encoder(
        trial=trial,
        sentence_encoder_output_dim=sentence_encoder.get_output_dim(),
        extra_dim=dialogue_extra_dim
    )

    sentence_encoder_output_dropout = trial.suggest_float(
        name='sentence_encoder_output_dropout', low=0.0, high=0.5
    )

    embedding_dropout = trial.suggest_float(name='embedding_dropout',
                                            low=0.0, high=0.5)
    dialogue_encoder_output_dropout = trial.suggest_float(
        name='dialogue_encoder_output_dropout', low=0.0, high=0.5
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

    return model

def create_jamo_cnn_model(trial: optuna.Trial,
                          vocab: Vocabulary) -> Model:
    token_embedding_dim = trial.suggest_categorical(
        name='token_embedding_dim', choices=[150, 300]
    )
    token_vocab_size = vocab.get_vocab_size('tokens')
    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders={
            'tokens': Embedding(embedding_dim=token_embedding_dim,
                                num_embeddings=token_vocab_size)
        }
    )

    dialogue_encoder = create_dialogue_cnn_encoder(
        trial=trial,
        token_embedding_dim=token_embedding_dim
    )

    embedding_dropout = trial.suggest_float(name='embedding_dropout',
                                            low=0.0, high=0.5)
    dialogue_encoder_output_dropout = trial.suggest_float(
        name='dialogue_encoder_output_dropout', low=0.0, high=0.5
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

    model = JamoCnnModel(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        dialogue_encoder=dialogue_encoder,
        discriminator=discriminator,
        embedding_dropout=embedding_dropout,
        dialogue_encoder_output_dropout=dialogue_encoder_output_dropout
    )

    return model

def create_token_transformer_model(trial: optuna.Trial,
                                   vocab: Vocabulary) -> Model:
    token_embedding_dim = trial.suggest_categorical(
        name='token_embedding_dim', choices=[256, 512]
    )
    token_vocab_size = vocab.get_vocab_size('tokens')
    text_field_embedder = TransformerEmbeddings(
        vocab_size=token_vocab_size,
        embedding_dim=token_embedding_dim,
        dropout=trial.suggest_float(name='transformer_embedding_dropout',
                                     low=0.0, high=0.5)
    )

    transformer_encoder_layers = create_transformer_encoder_layer(trial=trial,
                                                                  token_embedding_dim=token_embedding_dim)
    dialogue_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layers,
                                          num_layers=trial.suggest_categorical(
        name='transformer_encoder_nlayers', choices=[4, 6])
                                          )
    dialogue_encoder_output_dropout = trial.suggest_float(
        name='dialogue_encoder_output_dropout', low=0.0, high=0.5
    )

    discriminator = FeedForward(
        input_dim=token_embedding_dim,
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

    model = TokenTransformerModel(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        dialogue_encoder=dialogue_encoder,
        discriminator=discriminator,
        dialogue_encoder_output_dropout=dialogue_encoder_output_dropout
    )

    return model

def optimize(trial: optuna.Trial) -> float:
    if FLAGS.model_type == 'token':
        reader = TokenReader(maximum_dialogue_length=FLAGS.max_dialogue_length,
                             maximum_sentence_length=FLAGS.max_sentence_length)
    elif FLAGS.model_type == 'jamo_cnn':
        reader = JamoReader(maximum_dialogue_length=FLAGS.max_dialogue_length,
                            maximum_sentence_length=FLAGS.max_sentence_length)
    elif FLAGS.model_type == 'token_transformer':
        reader = TransformerReader()
    else:
        raise ValueError('Unknown model_type')

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

    if FLAGS.model_type == 'token':
        model = create_token_model(trial=trial,
                                   vocab=vocab)
    elif FLAGS.model_type == 'jamo_cnn':
        model = create_jamo_cnn_model(trial=trial,
                                      vocab=vocab)
    elif FLAGS.model_type == 'token_transformer':
        model = create_token_transformer_model(trial=trial,
                                               vocab=vocab)
    else:
        raise ValueError('Unknown model_type')
    if FLAGS.cuda_device >= 0:
        model = model.to(FLAGS.cuda_device)

    optimizer = Adam(params=model.parameters(),
                     lr=FLAGS.lr)
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
        params['validation_metric'] = FLAGS.val_metric
        json.dump(params, f, sort_keys=True, indent=4)
        logging.info(f'Params: {params}')
        logging.info(f'Saved params file to {params_path}')
    vocab_dir = save_dir / 'vocab'
    vocab.save_to_files(save_dir / 'vocab')
    logging.info(f'Saved vocab files to {vocab_dir}')

    tensorboard_writer = TensorboardWriter(
        serialization_dir=FLAGS.save_root_dir
    )

    lr_scheduler = LinearWithWarmup(optimizer=optimizer,
                                    num_epochs=FLAGS.num_epochs,
                                    num_steps_per_epoch=int(len(train_loader)),
                                    warmup_steps=FLAGS.warmup_steps)

    parser = ConfigParser()
    parser.read(FLAGS.config_path)

    @telegram_sender(token=parser.get('telegram', 'token'),
                     chat_id=parser.get('telegram', 'chat_id'))
    def train_notify(trainer):
        metrics = trainer.train()
        return metrics

    if FLAGS.val_metric == 'accuracy':
        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            patience=2,
            validation_metric='+accuracy',
            validation_data_loader=dev_loader,
            num_epochs=FLAGS.max_epochs,
            serialization_dir=save_dir,
            checkpointer=checkpointer,
            tensorboard_writer=tensorboard_writer,
            learning_rate_scheduler=lr_scheduler,
            cuda_device=FLAGS.cuda_device,
            grad_norm=5.0
        )
        metrics = train_notify(trainer)
        return metrics['best_validation_accuracy']
    elif FLAGS.val_metric == 'loss':
        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            patience=2,
            validation_metric='-loss',
            validation_data_loader=dev_loader,
            num_epochs=FLAGS.max_epochs,
            serialization_dir=save_dir,
            checkpointer=checkpointer,
            tensorboard_writer=tensorboard_writer,
            learning_rate_scheduler=lr_scheduler,
            cuda_device=FLAGS.cuda_device,
            grad_norm=5.0
        )
        metrics = train_notify(trainer)
        return metrics['best_validation_loss']

def validate_flags_encoder(flags_dict):
    if flags_dict['model_type'] == 'token':
        if flags_dict['sentence_encoder_type'] is None or flags_dict['dialogue_encoder_type'] is None:
            return False
    return True

def main(argv):
    if FLAGS.val_metric == 'accuracy':
        study = optuna.create_study(storage=FLAGS.optuna_storage,
                                    study_name=FLAGS.optuna_study_name,
                                    direction='maximize',
                                    load_if_exists=True)
    elif FLAGS.val_metric == 'loss':
        study = optuna.create_study(storage=FLAGS.optuna_storage,
                                    study_name=FLAGS.optuna_study_name,
                                    direction='minimize',
                                    load_if_exists=True)
    study.optimize(optimize, n_trials=FLAGS.optuna_num_trials)

    logging.info(f'The best trial is : \n{study.best_trial}')
    logging.info(f'The best value is : \n{study.best_value}')
    logging.info(f'The best parameters are : \n{study.best_params}')

    pickle.dump(study, open(FLAGS.save_root_dir + '/' + FLAGS.optuna_study_name + '.pkl', 'wb'))


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'model_type', 'min_token_count',
        'train_data_path', 'dev_data_path', 'save_root_dir',
        'optuna_study_name', 'config_path'
    ])
    flags.register_multi_flags_validator(
        flag_names=['model_type', 'sentence_encoder_type', 'dialogue_encoder_type'],
        multi_flags_checker=validate_flags_encoder,
        message='When model_type is "token", sentence_encoder_type and dialogue_encoder_type '
                'should be provided')

    app.run(main)
