import optuna
from absl import app, flags, logging
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('optuna_storage', default='sqlite:///optuna_studies.db',
                    help='URI to optuna storage')
flags.DEFINE_string('optuna_study_name', default=None,
                    help='Study name from which find the best trial')
flags.DEFINE_string('load_dir', default=None,
                    help='Root path to load optuna study')


def main(argv):
    logging.info(f'Load studies from {FLAGS.optuna_storage}')
    if not FLAGS.optuna_study_name:
        logging.info('No optuna_study_name is given.')
        logging.info('List of all available study names:')
        study_summaries = (
            optuna.get_all_study_summaries(storage=FLAGS.optuna_storage)
        )
        for s in study_summaries:
            start = (
                s.datetime_start.isoformat()
                if s.datetime_start is not None
                else None
            )
            logging.info('=' * 40)
            logging.info(f'Name: {s.study_name}')
            logging.info(f'Direction: {s.direction.name}')
            logging.info(f'Trials: {s.n_trials}')
            logging.info(f'Started at: {start}')
        logging.info('=' * 40)
    else:
        logging.info(f'Find the best trial from '
                     f'study {FLAGS.optuna_study_name}.')
        try:
            study = optuna.load_study(study_name=FLAGS.optuna_study_name,
                                      storage=FLAGS.optuna_storage)
        except:
            if not FLAGS.load_dir:
                pickle.load(open(FLAGS.load_dir + '/' + FLAGS.optuna_study_name + '.pkl', 'rb'))
            else:
                ValueError('No data to load')
        logging.info(f'Name: {study.study_name}')
        logging.info(f'Direction: {study.direction.name}')

        best_trial = study.best_trial
        logging.info(f'Best trial: Trial #{best_trial.number}')
        logging.info(f'Score: {best_trial.value}')
        logging.info('=' * 10 + ' Hyperparameters ' + '=' * 10)
        for k, v in best_trial.params.items():
            logging.info(f'{k}: {v}')


if __name__ == '__main__':
    app.run(main)
