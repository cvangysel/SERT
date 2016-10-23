import sys

from cvangysel import argparse_utils, io_utils, logging_utils, trec_utils

import argparse
import gzip
import json
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')
    parser.add_argument('--shard_size',
                        type=argparse_utils.positive_int, default=(1 << 14))

    parser.add_argument('reviews_file',
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

    with gzip.open(args.reviews_file, 'r') as f_reviews:
        for i, raw_line in enumerate(f_reviews):
            raw_line = raw_line.decode('utf8')

            review = json.loads(raw_line)

            product_id = review['asin']

            if product_list and product_id not in product_list:
                continue

            document_id = '{product_id}_{reviewer_id}_{review_time}'.format(
                product_id=product_id,
                reviewer_id=review['reviewerID'],
                review_time=review['unixReviewTime'])

            review_summary = review['summary']
            review_text = review['reviewText']

            product_document = '{0} \n{1}'.format(
                review_summary, review_text)

            product_document = io_utils.tokenize_text(product_document)

            document = ' '.join(product_document)

            writer.write_document(document_id, document)

            if (i + 1) % 1000 == 0:
                logging.info('Processed %d reviews.', i + 1)

    writer.close()

    logging.info('All done!')

if __name__ == "__main__":
    sys.exit(main())
