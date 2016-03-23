from __future__ import unicode_literals

import logging
import os
import subprocess


def configure_logging(args):
    loglevel = args.loglevel

    # Set logging level.
    numeric_log_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % loglevel.upper())

    logging.basicConfig(level=numeric_log_level)

    # Clear all default loggers.
    map(logging.getLogger().removeHandler, logging.getLogger().handlers[:])

    # Set-up log formatting.
    log_formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] [%(name)s] [%(levelname)s]  %(message)s')

    # Output to stderr.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(stream_handler)

    logging.info('Arguments: %s', args)
    logging.info('Git revision: %s', get_git_revision_hash())


def log_module_info(*modules):
    for module in modules:
        logging.info('%s version: %s (%s)',
                     module.__name__,
                     module.__version__,
                     module.__path__)


def get_git_revision_hash():
    try:
        proc = subprocess.Popen(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            cwd=os.path.dirname(os.path.realpath(__file__)))

        return proc.communicate()[0].strip()
    except:
        return None
