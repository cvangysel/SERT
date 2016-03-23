#!/usr/bin/env python

from __future__ import unicode_literals

import sys

from sert import argparse_utils, inference, io_utils, \
    logging_utils, math_utils

import argparse
import cPickle as pickle
import collections
import io
import logging
import numpy as np
import os
import re

#
# Main driver.
#


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--meta',
                        type=argparse_utils.existing_file_path, required=True)
    parser.add_argument('--model',
                        type=argparse_utils.existing_file_path, required=True)

    parser.add_argument('--topics',
                        type=argparse_utils.existing_file_path, nargs='+')

    parser.add_argument('--run_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    with open(args.model, 'rb') as f:
        unpickler = pickle.Unpickler(f)

        # Load model arguments and learned mapping.
        model_args, predict_fn = [unpickler.load() for _ in xrange(2)]

        # Load word representations.
        word_representations = unpickler.load()

    with open(args.meta, 'rb') as f:
        (data_args,
         words, tokens,
         expert_indices_inv, expert_assocs) = [
            pickle.load(f) for _ in xrange(5)]

    # Parse topic files.
    topics = parse_topics(
        map(lambda filename: open(filename, 'rb'), args.topics))

    model_name = os.path.basename(args.model)

    # Expert profiling.
    topics_per_expert = collections.defaultdict(list)

    # Expert finding.
    experts_per_topic = collections.defaultdict(list)

    def ranker_callback(topic_id, top_ranked_indices, top_ranked_values):
        for rank, (expert_internal_id, relevance) in enumerate(
                zip(top_ranked_indices, top_ranked_values)):
            expert_id = expert_indices_inv[expert_internal_id]

            # Expert profiling.
            topics_per_expert[expert_id].append((relevance, topic_id))

            # Expert finding.
            experts_per_topic[topic_id].append((relevance, expert_id))

    with open('{0}_debug'.format(args.run_out), 'w') as f_debug_out:
        result_callback = LogLinearCallback(
            args, model_args, tokens,
            f_debug_out,
            ranker_callback)

        batcher = inference.create(
            predict_fn, word_representations,
            model_args.batch_size, data_args.window_size, len(words),
            result_callback)

        logging.info('Batching queries using %s.', batcher)

        for q_id, (topic_id, terms) in enumerate(topics.iteritems()):
            if topic_id not in topics:
                logging.error('Topic "%s" not found in topic list.', topic_id)

                continue

            # Do not replace numeric tokens in queries.
            query_terms = parse_query(terms)

            query_tokens = []

            logging.debug('Query (%d/%d) %s: %s (%s)',
                          q_id + 1, len(topics),
                          topic_id, query_terms, terms)

            for term in query_terms:
                if term not in words:
                    logging.debug('Term "%s" is OOV.', term)

                    continue

                term_token = words[term].id

                query_tokens.append(term_token)

            if not query_tokens:
                logging.warning('Skipping query with terms "%s".', terms)

                continue

            batcher.submit(query_tokens, topic_id=topic_id)

        batcher.process()

    # Expert profiling.
    with io.open('{0}_ep'.format(args.run_out),
                 'w', encoding='utf8') as out_ep_run:
        write_run(model_name, topics_per_expert, out_ep_run)

    # Expert finding.
    with io.open('{0}_ef'.format(args.run_out),
                 'w', encoding='utf8') as out_ef_run:
        write_run(model_name, experts_per_topic, out_ef_run)

    logging.info('Saved run to %s.', args.run_out)


#
# Auxilary functions input/output.
#

def parse_topics(file_or_files,
                 max_topics=sys.maxint, delimiter=';', encoding='utf8'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and \
            not isinstance(file_or_files, tuple):
        file_or_files = [file_or_files]

    for f in file_or_files:
        assert isinstance(f, file)

        if f.encoding is not None and f.encoding != encoding:
            raise RuntimeError(
                'Encoding of file object different from expected '
                '(actual={0}, expected={1}).'.format(
                    f.encoding, encoding))

        for line in f:
            if not isinstance(line, unicode):
                line = line.decode(encoding)

            line = line.strip()
            if not line:
                continue

            topic_id, terms = line.strip().split(delimiter, 1)

            if topic_id in topics and (topics[topic_id] != terms):
                    logging.error('Duplicate topic "%s" (%s vs. %s).',
                                  topic_id,
                                  topics[topic_id],
                                  terms)

            topics[topic_id] = terms

            if max_topics > 0 and len(topics) >= max_topics:
                break

    return topics

remove_parentheses_re = re.compile(r'\((.*)\)')


def parse_query(unsplitted_terms):
    unsplitted_terms = remove_parentheses_re.sub(
        r'\1', unsplitted_terms.strip())
    unsplitted_terms = unicode(unsplitted_terms.replace('/', ' '))
    unsplitted_terms = unicode(unsplitted_terms.replace('-', ' '))

    return list(io_utils.token_stream(
        io_utils.lowercased_stream(
            io_utils.filter_non_latin_stream(
                io_utils.filter_non_alphanumeric_stream(
                    iter(unsplitted_terms)))), eos_chars=[]))


def write_run(model_name, data, out_f,
              max_objects_per_query=sys.maxint,
              skip_sorting=False):
    """
    Write a run to an output file.

    Parameters:
        - model_name: identifier of run.
        - data: dictionary mapping topic_id to object_assesments;
            object_assesments is an iterable (list or tuple) of
            (relevance, object_id) pairs.

            The object_assesments iterable is sorted by decreasing order.
        - out_f: output file stream.
        - max_objects_per_query: cut-off for number of objects per query.
    """
    for subject_id, object_assesments in data.iteritems():
        if not object_assesments:
            logging.warning('Received empty ranking for %s; ignoring.',
                            subject_id)

            continue

        # Probe types, to make sure everything goes alright.
        # assert isinstance(object_assesments[0][0], float) or \
        #     isinstance(object_assesments[0][0], np.float32)
        assert isinstance(object_assesments[0][1], basestring)

        if not skip_sorting:
            object_assesments = sorted(object_assesments, reverse=True)

        if max_objects_per_query < sys.maxint:
            object_assesments = object_assesments[:max_objects_per_query]

        if isinstance(subject_id, basestring):
            subject_id = subject_id.decode('utf8')

        for rank, (relevance, object_id) in enumerate(object_assesments):
            out_f.write(
                '{subject} Q0 {object} {rank} {relevance} '
                '{model_name}\n'.format(
                    subject=subject_id,
                    object=object_id.decode('utf8'),
                    rank=rank + 1,
                    relevance=relevance,
                    model_name=model_name))


#
# Ranker callbacks.
#

class Callback(object):

    def __init__(self, args, model_args, tokens,
                 f_debug_out,
                 rank_callback):
        self.args = args
        self.model_args = model_args

        self.tokens = tokens

        self.f_debug_out = f_debug_out

        self.rank_callback = rank_callback

        self.topic_projections = {}

    def __call__(self, payload, result, topic_id):
        assert topic_id not in self.topic_projections
        self.topic_projections[topic_id] = result.ravel()

        distribution = result

        logging.debug('Result of shape %s for topic "%s".',
                      distribution.shape, topic_id)

        self.process(payload, distribution, topic_id)

    def process(self, payload, distribution, topic_id):
        raise NotImplementedError()


class LogLinearCallback(Callback):

    def __init__(self, *args, **kwargs):
        super(LogLinearCallback, self).__init__(*args, **kwargs)

    def process(self, payload, distribution, topic_id):
        terms = map(lambda id: self.tokens[id], payload)
        term_entropies = compute_normalised_entropy(
            distribution, base=2)

        distribution = inference.aggregate_distribution(
            distribution, mode='product', axis=0)

        assert distribution.ndim == 1

        distribution /= distribution.sum()

        if not np.isclose(distribution.sum(), 1.0):
            logging.error('Encountered non-normalized '
                          'distribution for topic "%s" '
                          '(mass=%.10f).',
                          topic_id, distribution.sum())

        self.f_debug_out.write('Topic {0} {1}: {2}\n'.format(
            topic_id,
            math_utils.entropy(
                distribution, base=2, normalize=True),
            zip(terms, term_entropies)))

        ranked_indices = np.argsort(distribution)
        top_ranked_indices = ranked_indices[::-1]

        top_ranked_values = distribution[top_ranked_indices]

        self.rank_callback(topic_id, top_ranked_indices, top_ranked_values)


def compute_normalised_entropy(distribution, base=2):
    assert distribution.ndim == 2

    assert np.allclose(distribution.sum(axis=1), 1.0)

    entropies = [
        math_utils.entropy(distribution[i, :], base=base, normalize=True)
        for i in xrange(distribution.shape[0])]

    return entropies

if __name__ == "__main__":
    sys.exit(main())
