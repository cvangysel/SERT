from __future__ import unicode_literals

import Queue
import logging
import sys
import traceback


class WorkerFunction(object):
    """
        Decorator for multiprocessing worker functions which protects
        against failures raised within workers.

        Usage (top-level of your module):
            def worker_fn_(payload):
                pass

            worker_fn = multiprocessing_utils.WorkerFunction(worker_fn_)

        Afterwards, pass worker_fn to a multiprocessing.Pool instance.
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()

            logging.error(
                'Exception occured within worker: %s (%s); %s',
                exc_value, exc_type, traceback.format_tb(exc_traceback))

            raise


class QueueIterator(object):

    def __init__(self, pool, result_object, queue):
        self.pool = pool
        self.result_object = result_object
        self.queue = queue

        self.finished = False

        self.count = 0

    def next(self):
        while True:
            if self.result_object.ready() and not self.finished:
                logging.debug('All workers finished.')

                assert self.result_object.successful()

                worker_results = self.result_object.get()
                logging.debug('Retrieved results from workers: %s',
                              worker_results)

                if all(isinstance(result, int) for result in worker_results):
                    self.expected_number_items = sum(worker_results)

                self.finished = True

            queue_finished = self.queue.empty()

            # Safe-guarding code.
            if queue_finished and hasattr(self, 'expected_number_items'):
                logging.debug('Expected %d items; encountered %d.',
                              self.expected_number_items, self.count)

                if self.count > self.expected_number_items:
                    logging.error('Received more objects than expected.')

                    raise RuntimeError()

                queue_finished = (
                    queue_finished and
                    self.expected_number_items == self.count)

            if self.finished and queue_finished:
                logging.debug('Queue is empty.')

                self.pool.terminate()
                logging.debug('Pool terminated.')

                self.pool.join()
                logging.debug('Joined process pool thread.')

                self.queue.close()
                logging.debug('Result queue closed.')

                self.queue.join_thread()
                logging.debug('Joined result queue thread.')

                raise StopIteration()

            try:
                result = self.queue.get(block=False)
                self.count += 1

                return result
            except Queue.Empty:
                continue
