#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils, sklearn_utils, trec_utils
from sert import inference, math_utils, models

import argparse
import collections
import io
import logging
import numpy as np
import os
import operator
import pickle
import scipy
import scipy.spatial
import sklearn.neighbors

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

    parser.add_argument('--top',
                        type=argparse_utils.positive_int,
                        default=None)

    parser.add_argument('--run_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    with open(args.model, 'rb') as f:
        # Load model arguments and learned mapping.
        model_args, predict_fn = (pickle.load(f) for _ in range(2))

        # Load word representations.
        word_representations = pickle.load(f)

        try:
            entity_representations = pickle.load(f)
        except EOFError:
            entity_representations = None

    with open(args.meta, 'rb') as f:
        (data_args,
         words, tokens,
         entity_indices_inv, entity_assocs) = (
            pickle.load(f) for _ in range(5))

    # Parse topic files.
    topic_f = list(map(lambda filename: open(filename, 'r'), args.topics))
    topics = trec_utils.parse_topics(topic_f)
    [f_.close() for f_ in topic_f]

    model_name = os.path.basename(args.model)

    # Entity profiling.
    topics_per_entity = collections.defaultdict(list)

    # Entity finding.
    entities_per_topic = collections.defaultdict(list)

    def ranker_callback(topic_id, top_ranked_indices, top_ranked_values):
        for rank, (entity_internal_id, relevance) in enumerate(
                zip(top_ranked_indices, top_ranked_values)):
            entity_id = entity_indices_inv[entity_internal_id]

            # Entity profiling.
            topics_per_entity[entity_id].append((relevance, topic_id))

            # Entity finding.
            entities_per_topic[topic_id].append((relevance, entity_id))

    with open('{0}_debug'.format(args.run_out), 'w') as f_debug_out:
        if model_args.type == models.LanguageModel:
            result_callback = LogLinearCallback(
                args, model_args, tokens,
                f_debug_out,
                ranker_callback)
        elif model_args.type == models.VectorSpaceLanguageModel:
            result_callback = VectorSpaceCallback(
                entity_representations,
                args, model_args, tokens,
                f_debug_out,
                ranker_callback)

        batcher = inference.create(
            predict_fn, word_representations,
            model_args.batch_size, data_args.window_size, len(words),
            result_callback)

        logging.info('Batching queries using %s.', batcher)

        for q_id, (topic_id, terms) in enumerate(topics.items()):
            if topic_id not in topics:
                logging.error('Topic "%s" not found in topic list.', topic_id)

                continue

            # Do not replace numeric tokens in queries.
            query_terms = trec_utils.parse_query(terms)

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

    # Entity profiling.
    with io.open('{0}_ep'.format(args.run_out),
                 'w', encoding='utf8') as out_ep_run:
        trec_utils.write_run(model_name, topics_per_entity, out_ep_run)

    # Entity finding.
    with io.open('{0}_ef'.format(args.run_out),
                 'w', encoding='utf8') as out_ef_run:
        trec_utils.write_run(model_name, entities_per_topic, out_ef_run)

    logging.info('Saved run to %s.', args.run_out)


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

    def should_average_input(self):
        raise NotImplementedError()


class LogLinearCallback(Callback):

    def __init__(self, *args, **kwargs):
        super(LogLinearCallback, self).__init__(*args, **kwargs)

    def process(self, payload, distribution, topic_id):
        terms = list(map(lambda id: self.tokens[id], payload))
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

    def should_average_input(self):
        return False


class VectorSpaceCallback(Callback):

    def __init__(self, entity_representations, *args, **kwargs):
        super(VectorSpaceCallback, self).__init__(*args, **kwargs)

        logging.info(
            'Initializing k-NN for entity representations of shape %s.',
            entity_representations.shape)

        n_neighbors = self.args.top

        if n_neighbors is None:
            logging.warning(
                'Parameter k not set; defaulting to all entities (k=%d).',
                entity_representations.shape[0])
        elif n_neighbors > entity_representations.shape[0]:
            logging.warning(
                'Parameter k exceeds number of entities; '
                'defaulting to all entities (k=%d).',
                entity_representations.shape[0])

            n_neighbors = None

        self.entity_representation_distance = 'cosine'

        if self.entity_representation_distance == 'cosine':
            self.entity_representation_distance = 'euclidean'
            self.normalize_representations = True
        else:
            self.normalize_representations = False

        if self.normalize_representations:
            entity_repr_l2_norms = np.linalg.norm(
                entity_representations, axis=1)[:, np.newaxis]

            entity_representations /= entity_repr_l2_norms

            logging.debug('Term projections will be normalized.')

        self.entity_representations = entity_representations

        if n_neighbors:
            nn_impl = sklearn_utils.neighbors_algorithm(
                self.entity_representation_distance)

            logging.info('Using %s as distance metric in entity space '
                         'with NearestNeighbors %s implementation.',
                         self.entity_representation_distance, nn_impl)

            self.entity_neighbors = sklearn.neighbors.NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm=nn_impl,
                metric=self.entity_representation_distance)

            self.entity_neighbors.fit(entity_representations)
            self.entity_avg = entity_representations.mean(axis=1)

            logging.info('Entity k-NN params: %s',
                         self.entity_neighbors.get_params())
        else:
            logging.info('Using %s as distance metric in entity space.',
                         self.entity_representation_distance)

            self.entity_neighbors = None

    def query(self, centroids):
        if self.entity_neighbors is not None:
            distances, indices = self.entity_neighbors.kneighbors(centroids)

            return distances, indices
        else:
            pairwise_distances = scipy.spatial.distance.cdist(
                centroids, self.entity_representations,
                metric=self.entity_representation_distance)

            distances = np.sort(pairwise_distances, axis=1)
            indices = pairwise_distances.argsort(axis=1)\
                .argsort(axis=1).argsort(axis=1)

            return distances, indices

    def process(self, payload, result, topic_id):
        terms = list(map(lambda id: self.tokens[id], payload))

        term_projections = inference.aggregate_distribution(
            result, mode='identity', axis=0)

        if term_projections.ndim == 1:
            term_projections = term_projections.reshape(1, -1)

        _, entity_representation_size = term_projections.shape
        assert(entity_representation_size ==
               self.model_args.entity_representation_size)

        if self.normalize_representations:
            term_projections_l2_norm = \
                np.linalg.norm(term_projections, axis=1)[:, np.newaxis]
            term_projections /= term_projections_l2_norm

        logging.debug('Querying kneighbors for %s.', terms)

        distances, indices = self.query(term_projections)

        assert indices.shape[0] == term_projections.shape[0]

        candidates = collections.defaultdict(float)

        assert indices.shape[0] == 1

        for term in range(indices.shape[0]):
            term_indices = indices[term, :]

            for rank, candidate in enumerate(term_indices):
                matching_score = np.sum(
                    self.entity_representations[candidate, :] *
                    term_projections[term, :])

                if self.normalize_representations:
                    matching_score = (matching_score + 1.0) / 2.0

                candidates[candidate] += matching_score

        top_ranked_indices, top_ranked_values = \
            map(np.array, zip(
                *sorted(candidates.items(),
                        reverse=True,
                        key=operator.itemgetter(1))))

        self.rank_callback(topic_id, top_ranked_indices, top_ranked_values)

    def should_average_input(self):
        return True


def compute_normalised_entropy(distribution, base=2):
    assert distribution.ndim == 2

    assert np.allclose(distribution.sum(axis=1), 1.0)

    entropies = [
        math_utils.entropy(distribution[i, :], base=base, normalize=True)
        for i in range(distribution.shape[0])]

    return entropies

if __name__ == "__main__":
    sys.exit(main())
