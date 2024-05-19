from msmarco.download import msmarco_corpus_unzipped, DATA_ROOT
from msmarco.tokenizers import snowball_tokenizer

import pandas as pd
import pathlib
import csv
import sys
from searcharray import SearchArray

import logging
module_logger = logging.getLogger('searcharray.indexing')
# Send searcharray logs to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

module_logger.addHandler(ch)
module_logger.setLevel(logging.DEBUG)


def index_path(tokenizer=snowball_tokenizer):
    tokenizer_name = tokenizer.__name__
    return f"{DATA_ROOT}/msmarco_indexed_{tokenizer_name}.pkl"


def indexed_exists(tokenizer=snowball_tokenizer):
    return pathlib.Path(index_path(tokenizer=tokenizer)).exists()


csv.field_size_limit(sys.maxsize)

# Use csv iterator for memory efficiency
def index_msmarco(tokenizer=snowball_tokenizer):
    """Basic snowball stemmed msmarco indexing"""

    msmarco_unzipped_path = msmarco_corpus_unzipped()

    def csv_col_iter(msmarco_unzipped_path, col_no, num_rows=None):
        with open(msmarco_unzipped_path, "rt") as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row_no, row in enumerate(csv_reader):
                col = row[col_no]
                if num_rows is not None and row_no >= num_rows:
                    break
                yield col

    df = pd.DataFrame()
    # Read individually to not keep this DF in memory
    print("Saving ids")
    df['msmarco_id'] = pd.read_csv(msmarco_unzipped_path, delimiter="\t", usecols=[0], header=None)
    print("Getting URL")
    df['url'] = pd.read_csv(msmarco_unzipped_path, delimiter="\t", usecols=[1], header=None)
    print("Getting titles")
    df['title'] = pd.read_csv(msmarco_unzipped_path, delimiter="\t", usecols=[2], header=None)
    body_iter = csv_col_iter(msmarco_unzipped_path, 3)
    title_iter = csv_col_iter(msmarco_unzipped_path, 2)
    print("Indexing body")
    df['body_idx'] = SearchArray.index(body_iter, truncate=True, tokenizer=tokenizer)
    print("Indexing title")
    df['title_idx'] = SearchArray.index(title_iter, truncate=True, tokenizer=tokenizer)
    # Save to pickle
    df.to_pickle(index_path(tokenizer=tokenizer))


def indexed(tokenizer=snowball_tokenizer):
    if not indexed_exists(tokenizer=tokenizer):
        idx_path = index_path(tokenizer=tokenizer)
        print(f"Cannot find indexed file at {idx_path}")
        print("Indexing")
        index_msmarco()
    return pd.read_pickle(index_path(tokenizer=tokenizer))


if __name__ == "__main__":
    index_msmarco()
