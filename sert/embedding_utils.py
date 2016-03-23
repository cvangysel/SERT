import sys

import logging
import numpy as np
import os
import struct
import tempfile

if sys.version_info.major < 3:
    import cStringIO as StringIO
else:
    import io as StringIO


def get_binary_representations_info(filename):
    with open(filename, 'rb') as words_and_representations:
        vocabulary_size, vector_size = \
            map(int, words_and_representations.readline().strip().split())

        return vocabulary_size, vector_size


def load_binary_representations(filename, vocabulary=None):
    """Read vectors from a binary file."""
    logging.debug('Entered load_binary_representations')
    if vocabulary is not None:
        vocabulary = set(vocabulary)

    with open(filename, 'rb') as words_and_representations:
        vocabulary_size, vector_size = \
            map(int, words_and_representations.readline().strip().split())

        file_size = os.fstat(words_and_representations.fileno()).st_size

        last_reported_progress = 0

        while words_and_representations.tell() < file_size:
            logging.debug('Starting word')
            word_buffer = StringIO.StringIO()

            progress = int(
                100 * float(words_and_representations.tell()) / file_size)

            if progress % 10 == 0 and progress > last_reported_progress:
                logging.info(
                    'Reading file %s with %d words '
                    '(%d-dimensional): %d%% done.',
                    filename, vocabulary_size, vector_size, progress)

                last_reported_progress = progress

            while True:
                char = words_and_representations.read(1).decode()
                logging.debug("Reading character '{}'".format(char))

                if char == ' ' or not char:
                    break
                else:
                    word_buffer.write(str(char))

            word = word_buffer.getvalue().lower().strip()
            word_buffer.close()

            if not word and words_and_representations.tell() == file_size:
                # These were just some dangling whitespace characters at the
                # end of the file
                return

            buffer_size = struct.calcsize('f' * vector_size)

            representation_buffer = words_and_representations.read(buffer_size)

            if len(representation_buffer) < buffer_size:
                logging.error('Encountered end-of-file before reading '
                              'representation '
                              '(expected %d bytes, encountered %d bytes).',
                              buffer_size, len(representation_buffer))

                return

            # Ignore if not in vocabulary (if vocabulary is given).
            if vocabulary is not None and word not in vocabulary:
                continue

            representation = np.array(
                struct.unpack('f' * vector_size, representation_buffer))

            yield word, representation


def write_binary_representations(filename, words_and_representations):
    vector_size = None
    vocabulary_size = 0

    # Iterable words_and_representations is most likely not materialized yet.
    #
    # Therefore, we do not know the size of the vector representation and
    # neither the size of the vocabulary. We consume the iterable and write
    # everything to a temporary file.
    with tempfile.TemporaryFile() as tmp:
        for word, representation in words_and_representations:
            if vector_size is None:
                vector_size = len(representation)
            else:
                assert len(representation) == vector_size

            tmp.write(struct.pack('c' * len(word), *word))
            tmp.write(struct.pack('c', ' '))

            tmp.write(struct.pack('f' * vector_size, *representation))

            vocabulary_size += 1

        end_of_file = tmp.tell()

        # Reset temp file back to beginning.
        tmp.seek(0)

        with open(filename, 'wb') as f:
            f.write('%d %d\n' % (vocabulary_size, vector_size))

            while tmp.tell() < end_of_file:
                f.write(tmp.read(4096))

    logging.info('Wrote file %s with %d words (%d-dimensional).',
                 filename, vocabulary_size, vector_size)


def load_humanreadable_representations(filename):
    def _parse(line):
        return line[0], tuple(map(float, line[1:]))

    with open(filename, 'r') as f:
        for line in f:
            yield _parse(line.split(' '))


def write_humanreadable_representations(filename, words_and_representations):
    with open(filename, 'w') as f:
        for word, representation in words_and_representations:
            f.write('{0} {1}\n'.format(
                word, ' '.join(map(str, representation))))

    logging.info('Wrote file %s.', filename)
