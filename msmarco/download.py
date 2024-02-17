"""All msmarco URLs may need to be updated periodically from

https://microsoft.github.io/msmarco/Datasets.html
"""

import requests
import pathlib
import gzip


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


def msmarco_corpus_path():
    return "data/msmarco-docs.tsv.gz"


def msmarco_qrels_path():
    return "data/msmarco-doctrain-qrels.tsv.gz"


def msmarco_queries_path():
    return "data/msmarco-doctrain-queries.tsv.gz"


def msmarco_exists():
    corpus_path = pathlib.Path(msmarco_corpus_path())
    qr_path = pathlib.Path(msmarco_qrels_path())
    queries_path = pathlib.Path(msmarco_queries_path())
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
        # Ensure data directory
        pathlib.Path("data").mkdir(exist_ok=True)
        # Move to data directory
        path = url.split("/")[-1]
        pathlib.Path(path).rename(f"data/{path}")


def msmarco_download(force=False):
    if not msmarco_exists() or force:
        download_msmarco()
    return msmarco_corpus_path(), msmarco_qrels_path(), msmarco_queries_path()


def qrels():
    return msmarco_download()[1]


def corpus():
    return msmarco_download()[0]


def queries():
    return msmarco_download()[2]


def msmarco_corpus_unzipped():
    if not msmarco_exists():
        download_msmarco()
    path = msmarco_corpus_path()

    # Loop every .gz file in data and unzip
    msmarco_unzipped_path = 'data/msmarco-docs.tsv'
    msmarco_unzipped_path = pathlib.Path(msmarco_unzipped_path)

    if not msmarco_unzipped_path.exists():
        with gzip.open(path, 'rb') as f_in:
            with open(msmarco_unzipped_path, 'wb') as f_out:
                f_out.write(f_in.read())
    return msmarco_unzipped_path
