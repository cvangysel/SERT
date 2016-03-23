from __future__ import unicode_literals

import logging
import numpy as np


def create(predict_fn, word_representations,
           batch_size, window_size, vocabulary_size,
           result_callback):
    assert result_callback is not None

    instance_dtype = np.min_scalar_type(vocabulary_size - 1)
    logging.info('Instance elements will be stored using %s.', instance_dtype)

    batcher = WordBatcher(
        predict_fn,
        batch_size, window_size, instance_dtype,
        result_callback)

    return batcher


class WordBatcher(object):

    OVERFLOW, TRUNCATE = range(5, 7)

    def __init__(self, predict_fn,
                 batch_size, window_size, instance_dtype,
                 result_callback=None,
                 overflow_mode=OVERFLOW):
        self.predict_fn = predict_fn

        self.batch_size = batch_size
        self.window_size = window_size

        assert overflow_mode in (
            WordBatcher.OVERFLOW, WordBatcher.TRUNCATE)

        self.overflow_mode = overflow_mode

        logging.debug(
            'Using overflow mode "%s" for queries.',
            'truncate' if self.overflow_mode == WordBatcher.TRUNCATE
            else 'overflow')

        self.batch = np.zeros(
            (self.batch_size, self.window_size),
            dtype=instance_dtype)

        self.mask = np.zeros(
            (self.batch_size, self.window_size),
            dtype=np.int8)

        self._empty_batch()

        if result_callback is not None:
            assert hasattr(result_callback, '__call__')

        self.callback = result_callback

    def _empty_batch(self):
        self.batch[:, :] = 0
        self.mask[:, :] = 0

        self.num_used_instances = 0

        self.requests = []

    def process(self):
        if not self.requests:
            return

        logging.debug('Processing batch (batch size=%d, current batch=%d).',
                      self.batch_size, self.num_used_instances)

        results = self.predict_fn(self.batch, self.mask)

        logging.debug('Retrieved batch results %s.', results.shape)

        num_processed_instances = 0

        for i, (num_instances, payload, kwargs) in \
                enumerate(self.requests):
            result = results[
                num_processed_instances:
                num_processed_instances + num_instances, :, :]

            result = result.reshape((-1, result.shape[-1]))
            result = result[:len(payload), :]

            assert result.ndim == 2
            assert result.shape[0] == len(payload)

            self.callback(payload, result, **kwargs)

            num_processed_instances += num_instances

        self._empty_batch()

    def submit(self, query_tokens, **kwargs):
        assert len(query_tokens) > 0

        if len(query_tokens) > self.window_size and \
                self.overflow_mode == WordBatcher.TRUNCATE:
            logging.error('Truncated query "%s" as it exceeded '
                          'the window size.', query_tokens)

            query_tokens = query_tokens[:self.window_size]

        num_instances = len(query_tokens) / self.window_size
        if len(query_tokens) % self.window_size > 0:
            num_instances += 1

        logging.debug('Payload %s requires %d instances '
                      '(batch size=%d, current batch=%d).',
                      query_tokens, num_instances,
                      self.batch_size, self.num_used_instances)

        if num_instances > self.batch_size:
            raise RuntimeError()
        elif num_instances > \
                (self.batch_size - self.num_used_instances):
            self.process()

        self.requests.append((num_instances, query_tokens, kwargs))

        for i in xrange(num_instances):
            assert query_tokens

            instance_length = min(len(query_tokens), self.window_size)

            self.batch[self.num_used_instances, :instance_length] = \
                query_tokens[:instance_length]
            self.mask[self.num_used_instances, :instance_length] = 1

            query_tokens = query_tokens[instance_length:]

            self.num_used_instances += 1


def aggregate_distribution(distribution, mode, axis):
    if mode == 'sum':
        return np.mean(distribution, axis=axis)
    elif mode == 'product':
        return np.exp(np.sum(np.ma.log(distribution).filled(0), axis=axis))
    elif mode == 'last':
        return np.take(
            distribution, axis=axis, indices=distribution.shape[axis] - 1)
    elif mode == 'max':
        return np.max(distribution, axis=axis)
    elif mode == 'identity':
        return distribution
    else:
        raise NotImplementedError()
