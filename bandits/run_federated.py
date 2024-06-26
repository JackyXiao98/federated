# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python binary for bandits in TFF simulation."""
import collections
from collections.abc import Sequence
import os
from typing import Union

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from bandits import file_utils
from bandits import keras_optimizer_utils
from bandits import trainer


IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

# Experiment configuration
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    'bandits',
    (
        'The name of this experiment. Will beappend to  --root_output_dir to'
        ' separate experiment results. This name isoften generated by XM script'
        ' for each running task.'
    ),
)
_ROOT_OUTPUT_DIR = flags.DEFINE_string(
    'root_output_dir',
    '/tmp/bandits/',
    'Root directory for writing experiment output.',
)
_REPEAT = flags.DEFINE_integer(
    'repeat', '0', 'Repeat index to estimate the randomness of multiple runs.'
)

# Runtime configuration
_MAX_CONCURRENT_THREADS = flags.DEFINE_integer(
    'max_concurrent_threads',
    None,
    (
        'The maximum number of concurrent calls to a single computation in the'
        ' TFF CPP runtime. It is often used to avoid OOM on GPU for large'
        ' number of clients, and avoid parallelism bottleneck. If `None`, there'
        ' is no limit.'
    ),
)

# Task and algorithm
_TASK_TYPE = flags.DEFINE_enum(
    'task_type',
    None,
    [task_type.name.lower() for task_type in trainer.get_task_types()],
    'Which task (dataset and model) to use in the experiment.',
)
_BANDITS_TYPE = flags.DEFINE_enum(
    'bandits_type',
    None,
    [bandits_type.name.lower() for bandits_type in trainer.get_bandits_types()],
    'Which (bandits) algorithm to use in the experiment.',
)
_DIST_SHIFT_TYPE = flags.DEFINE_enum(
    'dist_shift_type',
    None,
    [dst.name.lower() for dst in trainer.get_distribution_types()],
    'If not None, decides the distribution shift setting.',
)
_POPULATION_CLIENT_SELECTION = flags.DEFINE_string(
    'population_client_selection',
    None,
    (
        'Selects a subset of clients; can be useful for distribution shift'
        ' settingsIn the format of "start_index-end_index".'
    ),
)

# Training configuration
_TOTAL_ROUNDS = flags.DEFINE_integer(
    'total_rounds', 10, 'Number of total training rounds.'
)
_CLIENTS_PER_ROUND = flags.DEFINE_integer(
    'clients_per_round', 10, 'How many clients to train on per round.'
)
_MAX_EXAMPLES_PER_CLIENT = flags.DEFINE_integer(
    'max_examples_per_client',
    None,
    (
        'Max number of training samples to use per client.'
        'Default to None, which will use all samples per client.'
    ),
)
_CLIENT_BATCH_SIZE = flags.DEFINE_integer(
    'client_batch_size', 16, 'Batch size used on the client.'
)
_CLIENT_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'client_shuffle_buffer_size',
    None,
    (
        'Size of buffer used to shuffle examples in client data '
        'input pipeline. No shuffling will be done if 0. The '
        'default, if None, is `10 * client_batch_size`.'
    ),
)
_CLIENT_EPOCHS_PER_ROUND = flags.DEFINE_integer(
    'client_epochs_per_round',
    1,
    'Number of epochs in the local client computation per round.',
)
_INITIAL_MODEL_PATH = flags.DEFINE_string(
    'initial_model_path',
    None,
    (
        'Path to a initial model of SavedModel format, which will be loaded '
        'by `tf.keras.models.load_model`. If None, no pretrained model is used.'
    ),
)

# Aggregator / DP
_AGG_TYPE = flags.DEFINE_enum(
    'aggregator_type',
    None,
    [agg_type.name.lower() for agg_type in trainer.get_aggregator_types()],
    (
        'Aggregator type: None, or dpsgd, or dpftrl. If `None`, use clipping'
        ' only or a normal weighted aggregator without noise.'
    ),
)
_UNCLIP_QUANTILE = flags.DEFINE_float(
    'unclip_quantile',
    None,
    (
        'Target quantile for adaptive clipping. If `None`, use clipping only '
        'or default robsut aggregator'
    ),
)
_CLIP_NORM = flags.DEFINE_float('clip_norm', None, 'Clip L2 norm.')
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier', None, 'Noise multiplier for DP algorithm.'
)
_ADAPTIVE_CLIPPING = flags.DEFINE_boolean(
    'adaptive_clipping', False, 'Adaptive clipping for default aggregator.'
)

# Eval and checkpoint
_ROUNDS_PER_EVAL = flags.DEFINE_integer(
    'rounds_per_eval',
    2,
    (
        'Evaluate the trained model on the validation dataset every '
        '`rounds_per_eval` rounds.'
    ),
)
_ROUNDS_PER_CHECKPOINT = flags.DEFINE_integer(
    'rounds_per_checkpoint',
    10,
    'Save checkpoints of server state every `rounds_per_checkpoint` rounds.',
)
_EVAL_BATCH_SIZE = flags.DEFINE_integer(
    'eval_batch_size', 256, 'Batch size used for online eval.'
)
_MAX_VAL_SAMPLES = flags.DEFINE_integer(
    'max_validation_samples', None, 'Max number of samples for validataion.'
)

# Optimizers
keras_optimizer_utils.define_optimizer_flags('client')
keras_optimizer_utils.define_optimizer_flags('server')

# Bandits
_EPSILON = flags.DEFINE_float(
    'bandits_epsilon', 0.2, 'Exploration of epsilon greedy bandits.'
)
_MU = flags.DEFINE_float(
    'bandits_mu',
    None,
    (
        'Exploration parameter mu of the FALCON bandit algorithm. If `None`, '
        'default to the number of possible actions.'
    ),
)
_GAMMA = flags.DEFINE_float(
    'bandits_gamma', 100, 'Exploration gamma of the FALCON bandit algorithm.'
)
_TEMPERATURE = flags.DEFINE_float(
    'bandits_temperature', 1.0, 'Temperature for softmax exploration.'
)
_DEPLOY_FREQ = flags.DEFINE_integer(
    'bandits_deploy_freq',
    100,
    (
        'Deploys the online training model for bandits'
        'inference every a few rounds.'
    ),
)

# Task Specific
_SO_VOCAB_SIZE = flags.DEFINE_integer(
    'stackoverflow_vocab_size',
    10000,
    'The vocabulary size of StackOveflow tag prediction.',
)
_SO_TAG_SIZE = flags.DEFINE_integer(
    'stackoverflow_tag_size', 50, 'The tag size of StackOveflow tag prediction.'
)

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _setup_tff_executor():
  """Configure TFF simulation runtime."""
  tff.backends.native.set_sync_local_cpp_execution_context(
      max_concurrent_computation_calls=_MAX_CONCURRENT_THREADS.value
  )


def _create_if_not_exists(path):
  """Creates a directory if it does not already exist."""
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _write_hparam_flags():
  """Save hyperparameter flags to be used together with saved metrics."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)
  hparam_dict = collections.OrderedDict(
      [(name, FLAGS[name].value) for name in HPARAM_FLAGS]
  )
  hparam_dict = keras_optimizer_utils.remove_unused_flags('client', hparam_dict)
  hparam_dict = keras_optimizer_utils.remove_unused_flags('server', hparam_dict)
  results_dir = os.path.join(
      _ROOT_OUTPUT_DIR.value, 'results', _EXPERIMENT_NAME.value
  )
  _create_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  file_utils.atomic_write_series_to_csv(hparam_dict, hparam_file)


def _check_positive(value: Union[int, float]):
  if value <= 0:
    raise ValueError(f'Got {value} for positive input.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(
        'Expected no command-line arguments, got: {}'.format(argv)
    )
  _setup_tff_executor()
  logging.info('Start training...')
  _write_hparam_flags()

  task_type_str = _TASK_TYPE.value
  if task_type_str is None:
    raise app.UsageError('Must specify --task_type')
  else:
    task_type = trainer.TaskType[task_type_str.upper()]

  bandits_type_str = _BANDITS_TYPE.value
  if bandits_type_str is None:
    raise app.UsageError('Must specify --bandits_type')
  else:
    bandits_type = trainer.BanditsType[bandits_type_str.upper()]

  dist_shift_type = _DIST_SHIFT_TYPE.value
  if dist_shift_type is not None:
    dist_shift_type = trainer.DistShiftType[dist_shift_type.upper()]

  agg_type = _AGG_TYPE.value
  if agg_type is not None:
    agg_type = trainer.AggregatorType[agg_type.upper()]

  program_state_manager, metrics_managers = trainer.configure_managers(
      _ROOT_OUTPUT_DIR.value, _EXPERIMENT_NAME.value
  )
  server_optimizer = keras_optimizer_utils.create_optimizer_fn_from_flags(
      'server'
  )
  client_optimizer = keras_optimizer_utils.create_optimizer_fn_from_flags(
      'client'
  )
  export_dir = os.path.join(
      _ROOT_OUTPUT_DIR.value, 'models', _EXPERIMENT_NAME.value
  )
  trainer.train_and_eval(
      task=task_type,
      bandits=bandits_type,
      distribution_shift=dist_shift_type,
      population_client_selection=_POPULATION_CLIENT_SELECTION.value,
      total_rounds=_TOTAL_ROUNDS.value,
      clients_per_round=_CLIENTS_PER_ROUND.value,
      rounds_per_eval=_ROUNDS_PER_EVAL.value,
      server_optimizer=server_optimizer,
      client_optimizer=client_optimizer,
      train_client_batch_size=_CLIENT_BATCH_SIZE.value,
      test_client_batch_size=_EVAL_BATCH_SIZE.value,
      train_client_epochs_per_round=_CLIENT_EPOCHS_PER_ROUND.value,
      train_shuffle_buffer_size=_CLIENT_SHUFFLE_BUFFER_SIZE.value,
      train_client_max_elements=_MAX_EXAMPLES_PER_CLIENT.value,
      bandits_epsilon=_EPSILON.value,
      bandits_temperature=_TEMPERATURE.value,
      bandits_mu=_MU.value,
      bandits_gamma=_GAMMA.value,
      bandits_deployment_frequency=_DEPLOY_FREQ.value,
      use_synthetic_data=False,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT.value,
      metrics_managers=metrics_managers,
      max_validation_samples=_MAX_VAL_SAMPLES.value,
      stackoverflow_vocab_size=_SO_VOCAB_SIZE.value,
      stackoverflow_tag_size=_SO_TAG_SIZE.value,
      initial_model_path=_INITIAL_MODEL_PATH.value,
      export_dir=export_dir,
      aggregator_type=agg_type,
      target_unclipped_quantile=_UNCLIP_QUANTILE.value,
      clip_norm=_CLIP_NORM.value,
      noise_multiplier=_NOISE_MULTIPLIER.value,
      adaptive_clipping=_ADAPTIVE_CLIPPING.value,
  )
  logging.info('Training completed.')


if __name__ == '__main__':
  flags.mark_flag_as_required('task_type')
  flags.mark_flag_as_required('bandits_type')
  app.run(main)
