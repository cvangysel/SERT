#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, embedding_utils, logging_utils
from sert import models

import argparse
import lasagne
import logging
import numpy as np
import os
import pickle
import scipy
import theano

#
# Main driver.
#

MODELS = {
    'loglinear': models.LanguageModel,
    'vectorspace': models.VectorSpaceLanguageModel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--data',
                        type=argparse_utils.existing_file_path, required=True)
    parser.add_argument('--meta',
                        type=argparse_utils.existing_file_path, required=True)

    parser.add_argument('--type', choices=MODELS, required=True)

    parser.add_argument('--iterations',
                        type=argparse_utils.positive_int, default=1)

    parser.add_argument('--batch_size',
                        type=argparse_utils.positive_int, default=1024)

    parser.add_argument('--word_representation_size',
                        type=argparse_utils.positive_int, default=300)
    parser.add_argument('--representation_initializer',
                        type=argparse_utils.existing_file_path, default=None)

    # Specific to VectorSpaceLanguageModel.
    parser.add_argument('--entity_representation_size',
                        type=argparse_utils.positive_int, default=None)
    parser.add_argument('--num_negative_samples',
                        type=argparse_utils.positive_int, default=None)
    parser.add_argument('--one_hot_classes',
                        action='store_true',
                        default=False)

    parser.add_argument('--regularization_lambda',
                        type=argparse_utils.ratio, default=0.01)

    parser.add_argument('--model_output', type=str, required=True)

    args = parser.parse_args()

    if args.entity_representation_size is None:
        args.entity_representation_size = args.word_representation_size

    args.type = MODELS[args.type]

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    logging_utils.log_module_info(theano, lasagne, np, scipy)

    # Load data.
    logging.info('Loading data from %s.', args.data)
    data_sets = np.load(args.data)

    if 'w_train' in data_sets and not args.ignore_weights:
        w_train = data_sets['w_train']
    else:
        logging.warning('No weights found in data set; '
                        'assuming uniform instance weighting.')

        w_train = np.ones(data_sets['x_train'].shape[0], dtype=np.float32)

    training_set = (data_sets['x_train'], data_sets['y_train'][()], w_train)
    validation_set = (data_sets['x_validate'], data_sets['y_validate'][()])

    logging.info('Training instances: %s (%s) %s (%s) %s (%s)',
                 training_set[0].shape, training_set[0].dtype,
                 training_set[1].shape, training_set[1].dtype,
                 training_set[2].shape, training_set[2].dtype)
    logging.info('Validation instances: %s (%s) %s (%s)',
                 validation_set[0].shape, validation_set[0].dtype,
                 validation_set[1].shape, validation_set[1].dtype)

    num_entities = training_set[1].shape[1]
    assert num_entities > 1

    if args.one_hot_classes:
        logging.info('Transforming y-values to one-hot values.')

        if not scipy.sparse.issparse(training_set[1]) or \
           not scipy.sparse.issparse(validation_set[1]):
            raise RuntimeError(
                'Argument --one_hot_classes expects sparse truth values.')

        y_train, (x_train, w_train) = sparse_to_one_hot_multiple(
            training_set[1], training_set[0], training_set[2])

        training_set = (x_train, y_train, w_train)

        y_validate, (x_validate,) = sparse_to_one_hot_multiple(
            validation_set[1], validation_set[0])

        validation_set = (x_validate, y_validate)

    logging.info('Loading meta-data from %s.', args.meta)
    with open(args.meta, 'rb') as f:
        # We do not load the remaining of the vocabulary.
        data_args, words, tokens = (pickle.load(f) for _ in range(3))

        vocabulary_size = len(words)

    representations = lasagne.init.GlorotUniform().sample(
        (vocabulary_size, args.word_representation_size))

    if args.representation_initializer:
        # This way of creating the dictionary ignores duplicate words in
        # the representation initializer.
        representation_lookup = dict(
            embedding_utils.load_binary_representations(
                args.representation_initializer, tokens))

        representation_init_count = 0

        for word, meta in words.items():
            if word.lower() in representation_lookup:
                representations[meta.id] = \
                    representation_lookup[word.lower()]

                representation_init_count += 1

        logging.info('Initialized representations from '
                     'pre-learned collection for %d words (%.2f%%).',
                     representation_init_count,
                     (representation_init_count /
                      float(len(words))) * 100.0)

    # Allow GC to clear memory.
    del words
    del tokens

    model_options = {
        'batch_size': args.batch_size,
        'window_size': data_args.window_size,
        'representations_init': representations,
        'regularization_lambda': args.regularization_lambda,
        'training_set': training_set,
        'validation_set': validation_set,
    }

    if args.type == models.LanguageModel:
        model_options.update(
            output_layer_size=num_entities)
    elif args.type == models.VectorSpaceLanguageModel:
        entity_representations = lasagne.init.GlorotUniform().sample(
            (num_entities, args.entity_representation_size))

        model_options.update(
            entity_representations_init=entity_representations,
            num_negative_samples=args.num_negative_samples)

    # Construct neural net.
    model = args.type(**model_options)

    train(model, args.iterations, args.model_output,
          abort_threshold=1e-5,
          early_stopping=False,
          additional_args=[args])


def sparse_to_one_hot_multiple(y, *matrices):
    """
    Convert sparse matrix to one-hot vector.

    Converts a 2-dimensional sparse matrix y to a 1-dimensional one-hot
    encoded vector. Similar to above, but supports multiple non-zero
    components in a single row of y.

    For every non-zero component in a row in y, the corresponding row in
    each additional passed matrix will be copied.
    Therefore, this implementation is less efficient than the one above.

    The returned matrices and vector are at least as large in their first
    dimension as the input matrices.
    """
    assert scipy.sparse.issparse(y), 'Matrix y should be sparse.'

    num_instances, num_classes = y.shape

    assert num_classes < (1 << 31), \
        'Number of classes should be encodable in 32-bit signed integer.'

    for matrix in matrices:
        assert isinstance(matrix, np.ndarray), \
            'Matrix {0} should be dense.'.format(repr(matrix))

        assert matrix.shape[0] == num_instances

    new_y = []
    new_matrices = [[] for _ in matrices]

    cx = y.tocoo()
    current_row = -1

    def emit(column_index):
        for i, matrix in enumerate(matrices):
            new_matrices[i].append(matrix[current_row])

        new_y.append(column_index)

    for i, j in zip(cx.row, cx.col):
        if i == current_row:
            emit(j)
        elif i == current_row + 1:
            current_row += 1

            emit(j)
        else:
            raise RuntimeError(
                'Every truth value should have at least '
                'one non-zero index.')

    new_y = np.array(new_y, dtype=np.int32)

    for i, new_matrix in enumerate(new_matrices):
        assert len(new_matrix) == new_y.size

        new_matrices[i] = np.array(new_matrix, dtype=matrices[i].dtype)

    return new_y, new_matrices


#
# Training driver.
#

def error_delta(error):
    if len(error) <= 1:
        return 0.0, 0.0
    else:
        absolute = error[-1] - error[-2]
        relative = absolute / float(error[-2])

        return absolute, relative


def train(model, num_epochs, output_path,
          abort_threshold=1e-5, early_stopping=False,
          additional_args=[]):
    assert isinstance(model, models.ModelInterface)
    assert isinstance(abort_threshold, float)

    error_means = {
        'training': [],
        'validation': [],
    }

    error_stddevs = {
        'training': [],
        'validation': [],
    }

    def compute_errors():
        # Compute errors once before any training occurs.
        train_error_mean, train_error_std = model.train_error()
        validation_error_mean, validation_error_std = model.validation_error()

        error_means['training'].append(train_error_mean)
        error_means['validation'].append(validation_error_mean)

        error_stddevs['training'].append(train_error_std)
        error_stddevs['validation'].append(validation_error_std)

    def dump_model(epoch):
        output_model_filename = '{0}_{1}.bin'.format(
            output_path, epoch)

        with open(output_model_filename, 'wb') as f:
            for obj in additional_args + list(model.get_state()):
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        model_size = os.path.getsize(output_model_filename)

        logging.info('Saved model "%s" (%d megabyte).',
                     output_model_filename, (model_size / 1024 / 1024))

    # Compute model errors at beginning.
    compute_errors()

    # Write the model before learning.
    dump_model(0)

    for epoch in range(1, num_epochs + 1):
        logging.info('Epoch %d.', epoch)

        num_batches, mean_cost = model.train()

        logging.info('Epoch %d: processed %d batches; average error=%f.',
                     epoch, num_batches, mean_cost)

        logging.info('Epoch %d: measuring training/validation error.', epoch)

        # Compute errors.
        compute_errors()

        logging.info('Training errors: %s; delta=%s',
                     list(zip(error_means['training'],
                              error_stddevs['training'])),
                     error_delta(error_means['training']))
        logging.info('Validation errors: %s; delta=%s',
                     list(zip(error_means['validation'],
                              error_stddevs['validation'])),
                     error_delta(error_means['validation']))

        dump_model(epoch=epoch)

        assert np.all(np.isfinite(error_means['training'][-1]))

        if early_stopping:
            assert np.all(np.isfinite(error_means['validation'][-1]))

            if error_means['validation'][-1] > error_means['validation'][-2]:
                logging.info('Validation error stopped decreasing; aborting.')

                return

        if len(error_means['training']) > 1 and \
                abs(error_means['training'][-1] -
                    error_means['training'][-2]) < abort_threshold:
            logging.error('No learning was performed during '
                          'the last iteration; aborting.')

            return

if __name__ == "__main__":
    sys.exit(main())
