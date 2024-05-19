"""All msmarco URLs may need to be updated periodically from

https://microsoft.github.io/msmarco/Datasets.html
"""
from msmarco.download import qrels, queries, msmarco_corpus_unzipped
import pandas as pd
import csv
import sys


csv.field_size_limit(sys.maxsize)


def judgments(num_judgments=None):
    queries_df = queries()
    qrels_df = qrels(nrows=num_judgments)    # Merge queries and qrels
    judgment_list = pd.merge(qrels_df, queries_df, on="query_id")
    return judgment_list


def grade_results(judgments, results) -> pd.DataFrame:
    # Merge judgments into results on doc_id, query_id
    labeled_results = pd.merge(results, judgments, on=["query_id", "msmarco_id"],
                               how="left")
    # Compute reciprical rank, 1/ rank, on each row
    labeled_results["reciprical_rank"] = 1 / labeled_results["rank"]
    labeled_results["grade"] = labeled_results["grade"].fillna(0)
    labeled_results.loc[labeled_results["grade"] != 1, "reciprical_rank"] = 0
    labeled_results = labeled_results.drop(columns=["query_y"])
    labeled_results.rename(columns={"query_x": "query"}, inplace=True)
    return labeled_results


def judge_queries(results_graded: pd.DataFrame, at=100) -> pd.DataFrame:
    results_graded = results_graded[results_graded['rank'] <= at]
    return results_graded.groupby('query')['reciprical_rank'].max()


def msmarco_of_ids(ids):
    def csv_col_gather(msmarco_unzipped_path, ids, id_col=0, num_rows=None):
        with open(msmarco_unzipped_path, "rt") as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row_no, row in enumerate(csv_reader):
                if row[id_col] in ids:
                    yield row
    df = []
    for row in csv_col_gather(msmarco_corpus_unzipped(), ids):
        df_row = {'msmarco_id': row[0], 'ideal_url': row[1], 'ideal_title': row[2], 'ideal_body': row[3]}
        df.append(df_row)
    return pd.DataFrame(df)
