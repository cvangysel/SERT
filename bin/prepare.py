#!/usr/bin/env python

from __future__ import unicode_literals

import sys

from sert import argparse_utils, logging_utils, multiprocessing_utils, io_utils

import multiprocessing
from nltk.corpus import stopwords

import argparse
import cPickle as pickle
import collections
import logging
import numpy as np
import os
import scipy
from scipy import sparse
import sklearn
import sklearn.cross_validation
import re

#
# Main driver.
#


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('document_paths',
                        type=argparse_utils.existing_file_path, nargs='+')

    parser.add_argument('--assoc_path',
                        type=argparse_utils.existing_file_path, required=True)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--vocabulary_min_count', type=int, default=2)
    parser.add_argument('--vocabulary_min_word_size', type=int, default=2)
    parser.add_argument('--vocabulary_max_size', type=int, default=65536)

    parser.add_argument('--remove_stopwords', type=str, default='nltk')

    parser.add_argument('--validation_set_ratio',
                        type=argparse_utils.ratio, default=0.01)

    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--overlapping', action='store_true', default=False)
    parser.add_argument('--stride',
                        type=argparse_utils.positive_int,
                        default=None)

    parser.add_argument('--no_shuffle', action='store_true', default=False)
    parser.add_argument('--no_padding', action='store_true', default=False)

    parser.add_argument('--meta_output',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)
    parser.add_argument('--data_output',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    logging_utils.log_module_info(np, scipy, sklearn)

    ignore_words = ['<doc>', '</doc>', '<text>', '</text>']
    if args.remove_stopwords == 'none':
        logging.info('Stopwords will be included in instances.')
    elif args.remove_stopwords == 'nltk':
        logging.info('Using NLTK stopword list.')

        ignore_words.extend(stopwords.words('english'))
        ignore_words.extend(stopwords.words('dutch'))
    elif os.path.exists(args.remove_stopwords):
        logging.info('Using custom stopword list (%s).', args.remove_stopwords)

        with open(args.remove_stopwords, 'r') as f:
            ignore_words.extend(
                filter(len, map(str.strip, map(str.lower, f.readlines()))))
    else:
        logging.error('Invalid stopword removal strategy "%s".',
                      args.remove_stopwords)

        return -1

    logging.info('Ignoring words: %s.', ignore_words)

    words, tokens = io_utils.extract_vocabulary(
        args.document_paths,
        min_count=args.vocabulary_min_count,
        max_vocab_size=args.vocabulary_max_size,
        min_word_size=args.vocabulary_min_word_size,
        num_workers=args.num_workers,
        ignore_tokens=ignore_words)

    logging.info('Loading document identifiers.')

    reader = TRECTextReader(args.document_paths)
    document_ids = reader.iter_document_ids(num_workers=args.num_workers)

    with open(args.assoc_path, 'r') as f_assocs:
        assocs = EntityDocumentAssociations(
            f_assocs,
            document_ids=document_ids)

    logging.info('Found %d unique entities.', len(assocs.entities))

    logging.info('Document-per-expert stats: %s',
                 map(lambda kv: (kv[0], len(kv[1])),
                     sorted(assocs.documents_per_entity.iteritems())))

    logging.info(
        'Expert-per-document association stats: %s',
        collections.Counter(
            map(len, assocs.entities_per_document.itervalues())).items())

    # Estimate the position in authorship distribution.
    num_associations_distribution = np.zeros(
        assocs.max_entities_per_document, dtype=np.int32)

    for association_length in (
            len(associated_experts)
            for associated_experts in
            assocs.entities_per_document.itervalues()):
        num_associations_distribution[association_length - 1] += 1

    logging.info('Number of associations distribution: %s',
                 num_associations_distribution)

    position_in_associations_distribution = np.cumsum(
        num_associations_distribution[::-1])[::-1]

    logging.info('Position in associations distribution: %s',
                 position_in_associations_distribution)

    instances_and_labels = []

    num_documents = 0
    num_non_associated_documents = 0

    documents_per_expert = collections.defaultdict(int)
    instances_per_expert = collections.defaultdict(int)
    instances_per_document = {}

    global_label_distribution = collections.defaultdict(float)

    if args.overlapping and args.stride is None:
        args.stride = 1
    elif args.overlapping and args.stride is not None:
        logging.error('Option --overlapping passed '
                      'concurrently with --stride.')

        return -1
    elif args.stride is None:
        logging.info('Defaulting stride to window size.')

        args.stride = args.window_size

    logging.info('Generating instances with stride %d.', args.stride)

    result_q = multiprocessing.Queue()

    pool = multiprocessing.Pool(
        args.num_workers,
        initializer=prepare_initializer,
        initargs=[result_q,
                  args, assocs.entities, assocs.entities_per_document,
                  position_in_associations_distribution,
                  tokens, words])

    max_document_length = 0

    worker_result = pool.map_async(
        prepare_worker, args.document_paths)

    # We will not submit any more tasks to the pool.
    pool.close()

    it = multiprocessing_utils.QueueIterator(
        pool, worker_result, result_q)

    num_labels = 0

    while True:
        try:
            result = next(it)
        except StopIteration:
            break

        num_documents += 1

        if result:
            document_id, \
                document_instances_and_labels, \
                document_label = result

            assert document_id not in instances_per_document

            num_instances_for_doc = len(document_instances_and_labels)
            instances_per_document[document_id] = num_instances_for_doc

            max_document_length = max(max_document_length,
                                      num_instances_for_doc)

            # For statistical purposes.
            for expert_id in assocs.entities_per_document[document_id]:
                documents_per_expert[expert_id] += 1

            # For statistical purposes.
            for expert_id in assocs.entities_per_document[document_id]:
                instances_per_expert[expert_id] += num_instances_for_doc
                num_labels += num_instances_for_doc

            # Aggregate.
            instances_and_labels.extend(document_instances_and_labels)

            # Some more accounting.
            for expert_id, mass in document_label.iteritems():
                global_label_distribution[expert_id] += \
                    num_instances_for_doc * mass
        else:
            num_non_associated_documents += 1

    # assert result_q.empty()

    logging.info('Global unnormalized distribution: %s',
                 global_label_distribution)

    logging.info(
        'Documents-per-indexed-expert stats (mean=%.2f, std_dev=%.2f): %s',
        np.mean(documents_per_expert.values()),
        np.std(documents_per_expert.values()),
        sorted(documents_per_expert.iteritems()))

    logging.info(
        'Instances-per-indexed-expert stats '
        '(mean=%.2f, std_dev=%.2f, min=%d, max=%d): %s',
        np.mean(instances_per_expert.values()),
        np.std(instances_per_expert.values()),
        np.min(instances_per_expert.values()),
        np.max(instances_per_expert.values()),
        sorted(instances_per_expert.iteritems()))

    logging.info(
        'Instances-per-document stats (mean=%.2f, std_dev=%.2f, max=%d).',
        np.mean(instances_per_document.values()),
        np.std(instances_per_document.values()),
        max_document_length)

    logging.info('Observed %d documents of which %d (ratio=%.2f) '
                 'are not associated with an expert.',
                 num_documents, num_non_associated_documents,
                 (float(num_non_associated_documents) / num_documents))

    training_instances_and_labels, validation_instances_and_labels = \
        sklearn.cross_validation.train_test_split(
            instances_and_labels, test_size=args.validation_set_ratio)

    num_training_instances = len(training_instances_and_labels)
    num_validation_instances = len(validation_instances_and_labels)

    num_instances = num_training_instances + num_validation_instances

    logging.info(
        'Processed %d instances; training=%d, validation=%d (ratio=%.2f).',
        num_instances,
        num_training_instances, num_validation_instances,
        (float(num_validation_instances) /
         (num_training_instances + num_validation_instances)))

    # Figure out if there are any experts with no instances, and
    # do not consider them during training.
    expert_indices = {}
    expert_indices_inv = {}

    for expert_id, num_instances in instances_per_expert.iteritems():
        if not num_instances:
            continue

        expert_index = len(expert_indices)

        expert_indices[expert_id] = expert_index
        expert_indices_inv[expert_index] = expert_id

    logging.info('Retained %d experts after instance creation.',
                 len(expert_indices))

    directories = map(os.path.dirname,
                      [args.meta_output, args.data_output])

    # Create those directories.
    [os.makedirs(directory) for directory in directories
     if not os.path.exists(directory)]

    # Dump vocabulary.
    with open(args.meta_output, 'wb') as f:
        for obj in (args, words, tokens,
                    expert_indices_inv, assocs.documents_per_entity):
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('Saved vocabulary to "%s".', args.meta_output)

    instance_dtype = np.min_scalar_type(len(words) - 1)
    logging.info('Instance elements will be stored using %s.', instance_dtype)

    data = {}

    x_train, y_train = instances_and_labels_to_arrays(
        training_instances_and_labels,
        args.window_size,
        expert_indices,
        instance_dtype,
        not args.no_shuffle)

    data['x_train'] = x_train
    data['y_train'] = y_train

    w_train = np.fromiter(
        (float(max_document_length) / instances_per_document[doc_id]
         for doc_id, _, _ in training_instances_and_labels),
        np.float32, len(training_instances_and_labels))

    assert w_train.shape == (x_train.shape[0],)

    data['w_train'] = w_train

    x_validate, y_validate = instances_and_labels_to_arrays(
        validation_instances_and_labels,
        args.window_size,
        expert_indices,
        instance_dtype,
        not args.no_shuffle)

    data['x_validate'] = x_validate
    data['y_validate'] = y_validate

    with open(args.data_output, 'wb') as f:
        np.savez(f, **data)

    logging.info('Saved data sets.')

    logging.info('Expert-per-document association stats: {0}'.format(
        collections.Counter(
            map(len, assocs.entities_per_document.itervalues())).items()))

    logging.info('Documents-per-expert stats: {0}'.format(
        collections.Counter(
            map(len, assocs.documents_per_entity.itervalues())).items()))

    logging.info('Done.')

#
# Utilities for parsing TRECtext and association files.
#


def _parse_trectext(iter, ignore_content=False):
    start_doc_re = re.compile(r'^<DOC>$')
    end_doc_re = re.compile(r'^</DOC>$')

    start_doc_hdr = re.compile(r'^<DOCHDR>$')
    end_doc_hdr = re.compile(r'^</DOCHDR>$')

    doc_id_re = re.compile(r'^<DOCNO>\s*(.*)\s*</DOCNO>$')

    current_document = None
    current_content = None

    for line in iter:
        doc_id_match = doc_id_re.match(line)

        if line.isspace():
            continue
        elif start_doc_re.match(line):
            assert current_document is None

            current_document = {
                'id': None,
                'header': [],
                'content': [],
            }

            current_content = current_document['content']
        elif end_doc_re.match(line):
            assert current_document is not None

            yield current_document['id'], current_document['content']

            current_document = None
            current_content = None
        elif start_doc_hdr.match(line):
            assert current_document is not None
            assert current_document['id'] is not None
            assert not current_document['content']

            current_content = current_document['header']
        elif end_doc_hdr.match(line):
            assert current_document is not None
            assert current_document['id'] is not None
            assert not current_document['content']

            current_content = current_document['content']
        elif doc_id_match:
            assert current_document is not None
            assert current_document['id'] is None

            current_document['id'] = doc_id_match.group(1)
        else:
            if current_document is None:
                logging.error(
                    'Encountered input outside of document context: %s', line)

                continue
            elif current_document['id'] is None:
                logging.error(
                    'Encountered input before document identifier: %s', line)

                continue
            elif current_content is None:
                logging.error(
                    'Encountered input within document without context: %s',
                    line)

                continue

            if ignore_content:
                continue

            line = line.strip()

            current_content.append(io_utils.filter_non_ascii(line))


def _iter_trectext_document_ids_worker(document_path):
    logging.info('Iterating over %s.', document_path)

    with open(document_path, 'r') as f:
        return [doc_id for doc_id, _ in _parse_trectext(f)]


class TRECTextReader(object):

    def __init__(self, document_paths):
        self.document_paths = document_paths

    def iter_document_ids(self, num_workers=1):
        document_ids = set()

        pool = multiprocessing.Pool(num_workers)

        for chunk_document_ids in pool.map(
                _iter_trectext_document_ids_worker, self.document_paths):
            document_ids.update(set(chunk_document_ids))

        return document_ids

    def iter_documents(self, replace_digits=True, strip_html=True):
        digit_regex = re.compile('\d+')

        for document_path in self.document_paths:
            logging.info('Iterating over %s.', document_path)

            with open(document_path, 'r') as f:
                for doc_id, text in _parse_trectext(f):
                    text = ' '.join(text)

                    if strip_html:
                        text = io_utils.strip_html(text)

                    if replace_digits:
                        text = digit_regex.sub('<num>', text)

                    yield doc_id, text


class EntityDocumentAssociations(object):

    def __init__(self, f, document_ids=None, max_unique_entities=False):
        self.entities = set()

        self.entities_per_document = collections.defaultdict(set)
        self.documents_per_entity = collections.defaultdict(set)

        self.max_entities_per_document = 0
        self.num_associations = 0

        for entity_id, document_id, _ in (
                assoc.strip().split() for assoc in f):
            if document_ids is not None and document_id not in document_ids:
                continue

            # If we only want to keep track of a maximum number of entities.
            if max_unique_entities:
                # Check if we already know the entity, if so, carry on.
                if entity_id in self.entities:
                    pass
                # If not, verify if we are still below the maximum. If not,
                # jump to next line.
                elif len(self.entities) >= max_unique_entities:
                    continue

            self.entities.add(entity_id)

            self.entities_per_document[document_id].add(entity_id)
            self.max_entities_per_document = max(
                self.max_entities_per_document,
                len(self.entities_per_document[document_id]))

            self.documents_per_entity[entity_id].add(document_id)

            self.num_associations += 1

#
# Worker functions.
#


def _candidate_centric_label(expert_ids):
    distribution = dict((expert_id, 1.0) for expert_id in expert_ids)

    return distribution


def prepare_initializer(
        _result_queue,
        _args, _experts, _document_assocs,
        _position_in_association_distribution,
        _tokens, _words):
    prepare_worker.result_queue = _result_queue

    prepare_worker.args = _args

    prepare_worker.experts = _experts
    prepare_worker.document_assocs = _document_assocs

    prepare_worker.position_in_associations_distribution = \
        _position_in_association_distribution

    prepare_worker.tokens = _tokens
    prepare_worker.words = _words

    prepare_worker.documents_per_expert_index = \
        collections.defaultdict(int)

    for expert_id in (
            expert_id for experts in
            prepare_worker.document_assocs.itervalues()
            for expert_id in experts):
        prepare_worker.documents_per_expert_index[expert_id] += 1

    logging.info('Worker initialized.')


def prepare_worker_(document_path):
    reader = TRECTextReader([document_path])
    num_documents = 0

    for doc_id, doc_text in reader.iter_documents(replace_digits=True,
                                                  strip_html=True):
        # Values to be returned.
        instances_and_labels = []

        if doc_id not in prepare_worker.document_assocs:
            logging.debug('Document "%s" does not exist in associations.',
                          doc_id)

            continue

        def _callback(num_yielded_windows, remaining_tokens):
            if not num_yielded_windows:
                logging.error('Document "%s" (%s) yielded zero instances; '
                              'remaining tokens: %s.',
                              doc_id, doc_text, remaining_tokens)

        padding_token = (
            '</s>' if not prepare_worker.args.no_padding else None)

        # Ignore end-of-sentence.
        windowed_word_stream = io_utils.windowed_translated_token_stream(
            io_utils.replace_numeric_tokens_stream(
                io_utils.token_stream(
                    io_utils.lowercased_stream(
                        io_utils.filter_non_latin_stream(
                            io_utils.filter_non_alphanumeric_stream(
                                iter(doc_text)))),
                    eos_chars=[])),
            prepare_worker.args.window_size,
            prepare_worker.words,
            stride=prepare_worker.args.stride,
            padding_token=padding_token,
            callback=_callback)

        # To determine the matrix indices of the experts associated with
        # the document.
        expert_ids = [expert_id for expert_id in
                      prepare_worker.document_assocs[doc_id]]

        label = _candidate_centric_label(expert_ids)
        partition_function = float(sum(label.values()))

        for index in label:
            label[index] /= partition_function

        for instance in windowed_word_stream:
            instances_and_labels.append((doc_id, instance, label))

        prepare_worker.result_queue.put(
            (doc_id, instances_and_labels, label))

        num_documents += 1

    return num_documents

prepare_worker = multiprocessing_utils.WorkerFunction(prepare_worker_)


#
# Utilities for packaging instances into arrays.
#


def instances_and_labels_to_arrays(instances,
                                   window_size,
                                   class_mapping,
                                   instance_dtype,
                                   shuffle):
    assert isinstance(instances, list)

    num_classes = len(class_mapping)

    if shuffle:
        logging.info('Shuffling instance and label pairs.')

        # Shuffle!
        np.random.shuffle(instances)
    else:
        logging.info('Instances are not shuffled.')

    num_instances = len(instances)

    # Unroll the instances.
    iterable = (element for _, instance, _ in instances
                for element in instance)

    logging.info('Constructing dense instance matrix.')

    x = np.fromiter(iterable,
                    dtype=instance_dtype,
                    count=num_instances * window_size)

    x = x.reshape((num_instances, window_size))

    assert x.shape == (num_instances, window_size)

    logging.info('Constructing sparse label matrix.')

    data = []
    row_ind = []
    col_ind = []

    for i, (_, _, instance_label) in enumerate(instances):
        assert isinstance(instance_label, dict)

        instance_col_ind, instance_data = zip(
            *sorted((class_mapping[expert_id], mass)
                    for expert_id, mass in instance_label.iteritems()))

        data.extend(instance_data)
        row_ind.extend([i] * len(instance_data))
        col_ind.extend(instance_col_ind)

    data = np.array(data, dtype=np.float32)

    y = sparse.csr_matrix(
        (data, (row_ind, col_ind)),
        shape=(num_instances, num_classes))

    return x, y

if __name__ == "__main__":
    sys.exit(main())
