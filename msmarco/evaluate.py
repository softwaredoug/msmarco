"""All msmarco URLs may need to be updated periodically from

https://microsoft.github.io/msmarco/Datasets.html
"""
from download import qrels, queries
import pandas as pd


def judgments():
    queries_df = pd.read_csv(queries(), delimiter="\t", header=None, names=["query_id", "query"])
    qrels_df = pd.read_csv(qrels(), delimiter=" ", header=None, names=["query_id", "q0", "doc_id", "grade"])

    # Merge queries and qrels
    judgment_list = pd.merge(qrels_df, queries_df, on="query_id")
    return judgment_list


def grade_results(judgments, results):
    # Merge judgments into results on doc_id, query_id
    labeled_results = pd.merge(results, judgments, on=["query_id", "doc_id"])
    # Compute reciprical rank, 1/ rank, on each row
    labeled_results["reciprical_rank"] = 1 / labeled_results["rank"]
    return labeled_results


def mrr(labeled_results):
    # Group by query_id
    grouped = labeled_results.groupby("query_id")
    return grouped["reciprical_rank"].max().mean()
