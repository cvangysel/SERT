#!/usr/bin/env python

from __future__ import unicode_literals

import sys

from sert import argparse_utils, embedding_utils, logging_utils, models

import argparse
import cPickle as pickle
import lasagne
import logging
import numpy as np
import os
import scipy
import theano

#
# Main driver.
#


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--data',
                        type=argparse_utils.existing_file_path, required=True)
    parser.add_argument('--meta',
                        type=argparse_utils.existing_file_path, required=True)

    parser.add_argument('--iterations',
                        type=argparse_utils.positive_int, default=1)

    parser.add_argument('--batch_size',
                        type=argparse_utils.positive_int, default=1024)

    parser.add_argument('--representation_size',
                        type=argparse_utils.positive_int, default=300)
    parser.add_argument('--representation_initializer',
                        type=argparse_utils.existing_file_path, default=None)

    parser.add_argument('--regularization_lambda',
                        type=argparse_utils.ratio, default=0.01)

    parser.add_argument('--model_output', type=str, required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    logging_utils.log_module_info(theano, lasagne, np, scipy)

    # Load data.
    logging.info('Loading data from %s.', args.data)
    data_sets = np.load(args.data)

    w_train = data_sets['w_train']

    training_set = (data_sets['x_train'], data_sets['y_train'][()], w_train)
    validation_set = (data_sets['x_validate'], data_sets['y_validate'][()])

    logging.info('Training instances: %s (%s) %s (%s) %s (%s)',
                 training_set[0].shape, training_set[0].dtype,
                 training_set[1].shape, training_set[1].dtype,
                 training_set[2].shape, training_set[2].dtype)
    logging.info('Validation instances: %s (%s) %s (%s)',
                 validation_set[0].shape, validation_set[0].dtype,
                 validation_set[1].shape, validation_set[1].dtype)

    num_classes = training_set[1].shape[1]
    assert num_classes > 1

    logging.info('Loading meta-data from %s.', args.meta)
    with open(args.meta, 'rb') as f:
        # We do not load the remaining of the vocabulary.
        data_args, words, tokens = (pickle.load(f) for _ in xrange(3))

        vocabulary_size = len(words)

    representations = lasagne.init.GlorotUniform().sample(
        (vocabulary_size, args.representation_size))

    if args.representation_initializer:
        # This way of creating the dictionary ignores duplicate words in
        # the representation initializer.
        representation_lookup = dict(
            embedding_utils.load_binary_representations(
                args.representation_initializer, tokens))

        representation_init_count = 0

        for word, meta in words.iteritems():
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
        'output_layer_size': num_classes,
        'representations_init': representations,
        'regularization_lambda': args.regularization_lambda,
        'training_set': training_set,
        'validation_set': validation_set,
    }

    # Construct neural net.
    model = models.LanguageModel(**model_options)

    train(model, args.iterations, args.model_output,
          abort_threshold=1e-5,
          early_stopping=False,
          additional_args=[args])


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

    for epoch in xrange(1, num_epochs + 1):
        logging.info('Epoch %d.', epoch)

        num_batches, mean_cost = model.train()

        logging.info('Epoch %d: processed %d batches; average error=%f.',
                     epoch, num_batches, mean_cost)

        logging.info('Epoch %d: measuring training/validation error.', epoch)

        # Compute errors.
        compute_errors()

        logging.info('Training errors: %s; delta=%s',
                     zip(error_means['training'],
                         error_stddevs['training']),
                     error_delta(error_means['training']))
        logging.info('Validation errors: %s; delta=%s',
                     zip(error_means['validation'],
                         error_stddevs['validation']),
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
