"""All msmarco URLs may need to be updated periodically from

https://microsoft.github.io/msmarco/Datasets.html
"""
from msmarco.download import qrels, queries
import pandas as pd


def judgments(num_judgments=None):
    queries_df = queries()
    qrels_df = qrels(nrows=num_judgments)    # Merge queries and qrels
    judgment_list = pd.merge(qrels_df, queries_df, on="query_id")
    return judgment_list


def grade_results(judgments, results) -> pd.DataFrame:
    # Merge judgments into results on doc_id, query_id
    labeled_results = pd.merge(results, judgments, on=["query_id", "msmarco_id"])
    # Compute reciprical rank, 1/ rank, on each row
    labeled_results["reciprical_rank"] = 1 / labeled_results["rank"]
    labeled_results = labeled_results.drop(columns=["query_y"])
    labeled_results.rename(columns={"query_x": "query"}, inplace=True)
    return labeled_results


def judge_queries(results_graded: pd.DataFrame, at=100) -> pd.DataFrame:
    results_graded = results_graded[results_graded['rank'] <= at]
    return results_graded.groupby('query')['reciprical_rank'].max()
