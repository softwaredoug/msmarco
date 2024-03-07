"""All msmarco URLs may need to be updated periodically from

https://microsoft.github.io/msmarco/Datasets.html
"""

import requests
import pathlib
import gzip
import pandas as pd


# Data root at home dir ~/.msmarco
DATA_ROOT = pathlib.Path.home() / ".msmarco"

# Create if doesn't exist
DATA_ROOT.mkdir(exist_ok=True)


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        print(f"Downloading {url}")
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {local_filename}")
    return local_filename


def _corpus_path():
    return f"{DATA_ROOT}/msmarco-docs.tsv.gz"


def _qrels_path():
    return f"{DATA_ROOT}/msmarco-doctrain-qrels.tsv.gz"


def _queries_path():
    return f"{DATA_ROOT}/msmarco-doctrain-queries.tsv.gz"


def msmarco_exists():
    corpus_path = pathlib.Path(_corpus_path())
    qr_path = pathlib.Path(_qrels_path())
    queries_path = pathlib.Path(_queries_path())
    return corpus_path.exists() and qr_path.exists() and queries_path.exists()


def download_msmarco():
    # Download to fixtures
    print("Downloading MSMARCO")

    urls = ["https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"]

    for url in urls:
        #  url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
        download_file(url)
        # Move to data directory
        path = url.split("/")[-1]
        pathlib.Path(path).rename(f"{DATA_ROOT}/{path}")


def msmarco_download(force=False):
    if not msmarco_exists() or force:
        download_msmarco()
    return _corpus_path(), _qrels_path(), _queries_path()


def qrels_path():
    return msmarco_download()[1]


def corpus_path():
    return msmarco_download()[0]


def queries_path():
    return msmarco_download()[2]


def queries() -> pd.DataFrame:
    return pd.read_csv(queries_path(), delimiter="\t", header=None, names=["query_id", "query"])


def corpus() -> pd.DataFrame:
    msmarco_unzipped_path = msmarco_corpus_unzipped()
    return pd.read_csv(msmarco_unzipped_path, delimiter="\t", header=None)


def _minimarco_path():
    return f"{DATA_ROOT}/minimarco.pkl"


def minimarco(size=None) -> pd.DataFrame:
    if pathlib.Path(_minimarco_path()).exists():
        return pd.read_pickle(_minimarco_path())

    queries_df = queries()
    qrels_df = qrels(nrows=size)    # Merge queries and qrels
    corpus_df = corpus()
    # Merge queries and qrels
    minimarco = pd.merge(qrels_df, queries_df, on="query_id")
    # Merge only corpus ids also in qrels
    minimarco.merge(corpus_df, on="msmarco_id", how="inner")
    minimarco.to_pickle(_minimarco_path())
    return minimarco


def qrels(nrows=10000) -> pd.DataFrame:
    return pd.read_csv(qrels_path(), delimiter=" ",
                       nrows=nrows,
                       header=None, names=["query_id", "q0", "msmarco_id", "grade"])


def msmarco_corpus_unzipped():
    if not msmarco_exists():
        download_msmarco()
    path = _corpus_path()

    # Loop every .gz file in data and unzip
    msmarco_unzipped_path = f"{DATA_ROOT}/msmarco-docs.tsv"
    msmarco_unzipped_path = pathlib.Path(msmarco_unzipped_path)

    if not msmarco_unzipped_path.exists():
        with gzip.open(path, 'rb') as f_in:
            with open(msmarco_unzipped_path, 'wb') as f_out:
                f_out.write(f_in.read())
    return msmarco_unzipped_path
