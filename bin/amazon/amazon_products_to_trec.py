import sys

from cvangysel import argparse_utils, io_utils, logging_utils, trec_utils

import argparse
import ast
import gzip
import logging
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')
    parser.add_argument('--shard_size',
                        type=argparse_utils.positive_int, default=(1 << 14))

    parser.add_argument('meta_file',
                        type=argparse_utils.existing_file_path)

    parser.add_argument('--product_list',
                        type=argparse_utils.existing_file_path,
                        default=None)

    parser.add_argument('--trectext_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    if args.product_list:
        with open(args.product_list, 'r') as f_product_list:
            product_list = set(
                product_id.strip()
                for product_id in f_product_list)

        logging.info('Only considering white list of %d products.',
                     len(product_list))
    else:
        product_list = None

    writer = trec_utils.ShardedTRECTextWriter(
        args.trectext_out, args.shard_size)

    department = ' '.join(
        os.path.basename(args.meta_file).split('.')[0]
        .split('_')[1:]).replace(' and ', ' & ')

    logging.info('Department: %s', department)

    with gzip.open(args.meta_file, 'r') as f_meta:
        for i, raw_line in enumerate(f_meta):
            raw_line = raw_line.decode('utf8')

            product = ast.literal_eval(raw_line)

            product_id = product['asin']

            if product_list and product_id not in product_list:
                continue

            if 'description' in product and 'title' in product:
                product_title = product['title']
                product_description = \
                    io_utils.strip_html(product['description'])

                product_document = '{0} \n{1}'.format(
                    product_title, product_description)

                product_document = io_utils.tokenize_text(product_document)

                logging.debug('Product %s has description of %d tokens.',
                              len(product_document))

                writer.write_document(
                    product_id, ' '.join(product_document))
            else:
                logging.debug(
                    'Filtering product %s due to missing description.',
                    product_id)

                continue

            if (i + 1) % 1000 == 0:
                logging.info('Processed %d products.', i + 1)

    writer.close()

    logging.info('All done!')

if __name__ == "__main__":
    sys.exit(main())
