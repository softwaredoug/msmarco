from msmarco.download import qrels, queries, corpus
import pandas as pd


def minimarco(size=None):
    queries_df = queries()
    qrels_df = qrels(nrows=size)    # Merge queries and qrels
    corpus_df = corpus()
    # Merge queries and qrels
    minimarco = pd.merge(qrels_df, queries_df, on="query_id")
    # Merge only corpus ids also in qrels
    minimarco.merge(corpus_df, on="msmarco_id", how="inner")
