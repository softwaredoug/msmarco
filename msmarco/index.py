from download import msmarco_unzipped
from searcharray import SearchArray
from tokenizers import snowball_tokenizer

import pandas as pd
import csv
import sys

import logging
module_logger = logging.getLogger('searcharray.indexing')
# Send searcharray logs to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

module_logger.addHandler(ch)
module_logger.setLevel(logging.DEBUG)


def index_msmarco():

    csv.field_size_limit(sys.maxsize)

    # Use csv iterator for memory efficiency
    def csv_col_iter(msmarco_unzipped_path, col_no, num_rows=None):
        with open(msmarco_unzipped_path, "rt") as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row_no, row in enumerate(csv_reader):
                col = row[col_no]
                if num_rows is not None and row_no >= num_rows:
                    break
                yield col

    msmarco_unzipped_path = msmarco_unzipped()

    body_iter = csv_col_iter(msmarco_unzipped_path, 3)
    title_iter = csv_col_iter(msmarco_unzipped_path, 2)
    df = pd.DataFrame()
    print("Indexing body")
    df['body_snowball'] = SearchArray.index(body_iter, truncate=True, tokenizer=snowball_tokenizer)
    print("Indexing title")
    df['title_snowball'] = SearchArray.index(title_iter, truncate=True, tokenizer=snowball_tokenizer)
    print("Saving ids")
    # df['msmarco_id'] = pd.read_csv(msmarco_unzipped_path, delimiter="\t", usecols=[0], header=None)
    print("Getting URL")
    # df['msmarco_id'] = pd.read_csv(msmarco_unzipped_path, delimiter="\t", usecols=[1], header=None)
    # Save to pickle
    df.to_pickle("data/msmarco_indexed.pkl")


if __name__ == "__main__":
    index_msmarco()
