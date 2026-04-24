# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run training of one or more algorithmic tasks from CLRS."""

import functools
from datetime import datetime
import os
import shutil
from typing import Any, Dict, List, Optional

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf

from clrs._src.multi_sol.evaluation import dispatch as multisol_dispatch
from clrs._src.multi_sol.evaluation import reporting as multisol_reporting


flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_list(
    'test_lengths',
    ['-1'],
    'Test sizes to use. -1 uses benchmark test dataset.')
flags.DEFINE_integer(
    'test_length',
    None,
    'Legacy alias for single test size; overrides --test_lengths when set.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_gmpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')
flags.DEFINE_enum(
    'evaluation_profile',
    'default',
    ['default', 'sampling'],
    'Evaluation profile: default CLRS metrics or multi-solution sampling plugins.')
flags.DEFINE_boolean(
    'save_sampling_artifacts',
    False,
    'If true and evaluation_profile=sampling, persist plugin artifacts in results/.')
flags.DEFINE_string(
    'sampling_artifact_prefix',
    'sampling_eval',
    'Filename prefix for saved sampling artifacts.')
flags.DEFINE_string(
    'run_dir',
    '',
    'Output directory for this run. Default: results/<timestamp>/.')
flags.DEFINE_string(
    'filename',
    '',
    'Legacy-compatible base filename for experiment outputs.')
flags.DEFINE_boolean(
    'results_df',
    False,
    'Collect train/validation metric rows for CSV export.')
flags.DEFINE_boolean(
    'save_df',
    False,
    'Persist collected metric rows to CSV (legacy-compatible behavior).')
flags.DEFINE_string(
    'results_df_filename',
    '',
    'Optional CSV filename for --save_df (default: score-results).')
flags.DEFINE_boolean(
    'save_model_to_file',
    False,
    'Persist full model params/optimizer state to a permanent .pkl file.')
flags.DEFINE_string(
    'model_output_path',
    '',
    'Output path for --save_model_to_file (default derived from --filename).')
flags.DEFINE_boolean(
    'validate_distributions',
    False,
    'Enable legacy distribution-validation side effects in extension evaluators.')
flags.DEFINE_integer(
    'NSE',
    25,
    'Legacy-compatible number of extracted solutions for distribution validation.')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray  # pytype: disable=attribute-error
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def create_samplers(
    rng,
    train_lengths: List[int],
    *,
    algorithms: Optional[List[str]] = None,
    val_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
):
  """Create samplers for training, validation and testing.

  Args:
    rng: Numpy random state.
    train_lengths: list of training lengths to use for each algorithm.
    algorithms: list of algorithms to generate samplers for. Set to
        FLAGS.algorithms if not provided.
    val_lengths: list of lengths for validation samplers for each algorithm. Set
        to maxumim training length if not provided.
    test_lengths: list of lengths for test samplers for each algorithm. Set to
        [-1] to use the benchmark dataset if not provided.
    train_batch_size: batch size for training samplers.
    val_batch_size: batch size for validation samplers.
    test_batch_size: batch size for test samplers.

  Returns:
    Tuple of:
      train_samplers: list of samplers for training.
      val_samplers: list of samplers for validation.
      val_sample_counts: list of sample counts for validation.
      test_samplers: list of samplers for testing.
      test_sample_counts: list of sample counts for testing.
      spec_list: list of specs for each algorithm.

  """

  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  algorithms = algorithms or FLAGS.algorithms
  for algo_idx, algorithm in enumerate(algorithms):
    # Set the training lengths for the current algorithm.
    current_algo_train_lengths = train_lengths

     # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):
      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(current_algo_train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        current_algo_train_lengths = [max_length]
        if FLAGS.chunked_training:
          current_algo_train_lengths = current_algo_train_lengths * len(
              current_algo_train_lengths
          )

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(
          sizes=current_algo_train_lengths,
          split='train',
          batch_size=train_batch_size,
          multiplier=-1,
          randomize_pos=FLAGS.random_pos,
          chunked=FLAGS.chunked_training,
          sampler_kwargs=sampler_kwargs,
          **common_sampler_args,
      )
      train_sampler, _, _ = make_multi_sampler(**train_args)

      algo_settings = clrs.CLRS_30_ALGS_SETTINGS.get(
          algorithm, {'num_samples_multiplier': 1})
      mult = algo_settings['num_samples_multiplier']
      val_args = dict(
          sizes=val_lengths or [np.amax(current_algo_train_lengths)],
          split='val',
          batch_size=val_batch_size,
          multiplier=2 * mult,
          randomize_pos=FLAGS.random_pos,
          chunked=False,
          sampler_kwargs=sampler_kwargs,
          **common_sampler_args,
      )
      val_sampler, val_samples, _ = make_multi_sampler(**val_args)

      test_args = dict(sizes=test_lengths or [-1],
                       split='test',
                       batch_size=test_batch_size,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def _resolve_test_lengths() -> List[int]:
  if FLAGS.test_length is not None:
    return [FLAGS.test_length]
  return [int(x) for x in FLAGS.test_lengths]


def _legacy_sampling_requested() -> bool:
  return bool(FLAGS.filename) or FLAGS.validate_distributions or FLAGS.NSE != 25


def _resolve_run_dir() -> str:
  if FLAGS.run_dir:
    run_dir = FLAGS.run_dir
  else:
    run_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
  os.makedirs(run_dir, exist_ok=True)
  return run_dir


def _effective_evaluation_profile() -> str:
  if FLAGS.evaluation_profile != 'default':
    return FLAGS.evaluation_profile
  if _legacy_sampling_requested():
    return 'sampling'
  return 'default'


def _extension_eval_kwargs(split: str, run_dir: str) -> Dict[str, Any]:
  kwargs = {
      'vd_flag': FLAGS.validate_distributions,
      'NSE': FLAGS.NSE,
      'output_dir': run_dir,
  }
  if split == 'test' and FLAGS.filename:
    kwargs['filename'] = FLAGS.filename
  return kwargs


def _sampling_report_sink(split: str, profile: str, run_dir: str):
  if split != 'test' or profile != 'sampling':
    return None
  if FLAGS.filename:
    return functools.partial(
        multisol_reporting.save_csv_report,
        output_dir=run_dir,
        timestamped=False,
    )
  if FLAGS.save_sampling_artifacts:
    return functools.partial(
        multisol_reporting.save_pickle_report,
        output_dir=run_dir,
        timestamped=True,
    )
  return None


def _default_results_df_filename() -> str:
  return FLAGS.results_df_filename or 'score-results'


def _default_model_output_path(run_dir: str) -> str:
  if FLAGS.model_output_path:
    return FLAGS.model_output_path
  if FLAGS.filename:
    return os.path.join(run_dir, f'{FLAGS.filename}_model.pkl')
  return os.path.join(run_dir, 'eval_model.pkl')


def _resolve_checkpoint_path(run_dir: str) -> str:
  checkpoint_flag = FLAGS['checkpoint_path']
  if checkpoint_flag.present:
    checkpoint_path = FLAGS.checkpoint_path
  else:
    checkpoint_path = os.path.join(run_dir, 'checkpoints')
  os.makedirs(checkpoint_path, exist_ok=True)
  return checkpoint_path


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]
  test_lengths = _resolve_test_lengths()
  run_dir = _resolve_run_dir()
  checkpoint_path = _resolve_checkpoint_path(run_dir)
  effective_profile = _effective_evaluation_profile()
  logging.info('Run output directory: %s', run_dir)
  logging.info('Checkpoint directory: %s', checkpoint_path)
  if effective_profile != FLAGS.evaluation_profile:
    logging.info(
        'Using compatibility evaluation profile "%s" (requested "%s").',
        effective_profile,
        FLAGS.evaluation_profile,
    )
  collect_results_df = FLAGS.results_df or FLAGS.save_df
  metric_rows = []

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))

  # Create samplers
  (
      train_samplers,
      val_samplers,
      val_sample_counts,
      test_samplers,
      test_sample_counts,
      spec_list,
  ) = create_samplers(
      rng=rng,
      train_lengths=train_lengths,
      algorithms=FLAGS.algorithms,
      val_lengths=[np.amax(train_lengths)],
      test_lengths=test_lengths,
      train_batch_size=FLAGS.batch_size,
  )

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model

  # Training loop.
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  train_loss_windows = [[] for _ in FLAGS.algorithms]
  step = 0
  next_eval = 0
  # Make sure scores improve on first step, but not overcome best score
  # until all algos have had at least one evaluation.
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0

  while step < FLAGS.train_steps:
    feedback_list = [next(t) for t in train_samplers]

    # Initialize model.
    if step == 0:
      all_features = [f.features for f in feedback_list]
      if FLAGS.chunked_training:
        # We need to initialize the model with samples of all lengths for
        # all algorithms. Also, we need to make sure that the order of these
        # sample sizes is the same as the order of the actual training sizes.
        all_length_features = [all_features] + [
            [next(t).features for t in train_samplers]
            for _ in range(len(train_lengths))]
        train_model.init(all_length_features[:-1], FLAGS.seed + 1)
      else:
        train_model.init(all_features, FLAGS.seed + 1)

    # Training step.
    for algo_idx in range(len(train_samplers)):
      feedback = feedback_list[algo_idx]
      rng_key, new_rng_key = jax.random.split(rng_key)
      if FLAGS.chunked_training:
        # In chunked training, we must indicate which training length we are
        # using, so the model uses the correct state.
        length_and_algo_idx = (length_idx, algo_idx)
      else:
        # In non-chunked training, all training lengths can be treated equally,
        # since there is no state to maintain between batches.
        length_and_algo_idx = algo_idx
      cur_loss = train_model.feedback(rng_key, feedback, length_and_algo_idx)
      rng_key = new_rng_key
      if collect_results_df:
        train_loss_windows[algo_idx].append(float(cur_loss))

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      logging.info('Algo %s step %i current loss %f, current_train_items %i.',
                   FLAGS.algorithms[algo_idx], step,
                   cur_loss, current_train_items[algo_idx])

    # Periodically evaluate model
    if step >= next_eval:
      eval_model.params = train_model.params
      for algo_idx in range(len(train_samplers)):
        common_extras = {'examples_seen': current_train_items[algo_idx],
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}

        # Validation info.
        new_rng_key, rng_key = jax.random.split(rng_key)
        val_stats = multisol_dispatch.evaluate_with_registry(
            algorithm_name=FLAGS.algorithms[algo_idx],
            split='val',
            profile=effective_profile,
            sampler=val_samplers[algo_idx],
            predict_fn=functools.partial(
                eval_model.predict, algorithm_index=algo_idx),
            sample_count=val_sample_counts[algo_idx],
            rng_key=new_rng_key,
            extras=common_extras,
            artifact_prefix=FLAGS.sampling_artifact_prefix,
            save_artifacts=False,
            fallback_eval_fn=collect_and_eval,
            extension_kwargs=_extension_eval_kwargs(split='val', run_dir=run_dir),
            report_sink=_sampling_report_sink(
                split='val', profile=effective_profile, run_dir=run_dir),
        )
        logging.info('(val) algo %s step %d: %s',
                     FLAGS.algorithms[algo_idx], step, val_stats)
        val_scores[algo_idx] = val_stats['score']
        if collect_results_df:
          mean_train_loss = (
              float(np.mean(train_loss_windows[algo_idx]))
              if train_loss_windows[algo_idx]
              else float('nan')
          )
          metric_rows.append({
              'Algorithm': FLAGS.algorithms[algo_idx],
              'Train KlDiv': mean_train_loss,
              'Mean 1-abs(error)': float(val_stats['score']),
              'Num Steps': int(step),
              'Examples Seen': int(current_train_items[algo_idx]),
          })
          train_loss_windows[algo_idx] = []

      next_eval += FLAGS.eval_every

      # If best total score, update best checkpoint.
      # Also save a best checkpoint on the first step.
      msg = (f'best avg val score was '
             f'{best_score/len(FLAGS.algorithms):.3f}, '
             f'current avg val score is {np.mean(val_scores):.3f}, '
             f'val scores are: ')
      msg += ', '.join(
          ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
      if (sum(val_scores) > best_score) or step == 0:
        best_score = sum(val_scores)
        logging.info('Checkpointing best model, %s', msg)
        train_model.save_model('best.pkl')
      else:
        logging.info('Not saving new best model, %s', msg)

    step += 1
    length_idx = (length_idx + 1) % len(train_lengths)

  if FLAGS.train_steps == 0 and eval_model.params is None:
    logging.info('No training steps requested, evaluation only. Initialising model...')
    eval_model.init([next(t).features for t in val_samplers], FLAGS.seed + 1)
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

  for algo_idx in range(len(train_samplers)):
    common_extras = {'examples_seen': current_train_items[algo_idx],
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}

    new_rng_key, rng_key = jax.random.split(rng_key)
    test_stats = multisol_dispatch.evaluate_with_registry(
        algorithm_name=FLAGS.algorithms[algo_idx],
        split='test',
        profile=effective_profile,
        sampler=test_samplers[algo_idx],
        predict_fn=functools.partial(eval_model.predict, algorithm_index=algo_idx),
        sample_count=test_sample_counts[algo_idx],
        rng_key=new_rng_key,
        extras=common_extras,
        artifact_prefix=FLAGS.sampling_artifact_prefix,
        save_artifacts=FLAGS.save_sampling_artifacts,
        fallback_eval_fn=collect_and_eval,
        extension_kwargs=_extension_eval_kwargs(split='test', run_dir=run_dir),
        report_sink=_sampling_report_sink(
            split='test', profile=effective_profile, run_dir=run_dir),
    )
    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

  if FLAGS.save_df and collect_results_df:
    multisol_reporting.save_csv_report(
        metric_rows,
        _default_results_df_filename(),
        output_dir=run_dir,
        timestamped=False,
    )

  if FLAGS.save_model_to_file:
    eval_model.save_model_to_permanent_file(_default_model_output_path(run_dir))

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
