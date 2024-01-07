from pathlib import Path

import nltk
import pandas as pd
from retriv import SparseRetriever

Datasets = {
    "walmart-amazon": Path("data/blocking/walmart-amazon_homo/"),
    "amazon-google": Path("data/blocking/amazon-google/"),
    "dblp-scholar": Path("data/blocking/dblp-scholar"),
    "dblp-acm": Path("data/blocking/dblp-acm"),
    "abt-buy": Path("data/blocking/abt-buy_heter"),
    # "wdc":
}

nltk.download = lambda *args, **kwargs: None


def generate_docs(df: pd.DataFrame):
    for idx, row in df.iterrows():
        yield {
            "id": idx,
            "text": " ".join(map(str, row.values)),
        }


def blocking(
    dataset: str,
    dfs: list[pd.DataFrame],
    matches: set[tuple],
    K: int = 20,
):
    try:
        retriever = SparseRetriever.load(f"{dataset}-index")
        raise FileNotFoundError
    except FileNotFoundError:
        retriever = SparseRetriever(index_name=f"{dataset}-index")
        retriever = retriever.index(
            generate_docs(dfs[-1]),
            show_progress=True,
        )
        retriever.save()

    queries = list(generate_docs(dfs[0]))
    candidates = retriever.bsearch(queries, show_progress=True)
    candidates_k = list()
    for lc, v in candidates.items():
        for rc in sorted(v, key=v.get, reverse=True)[:K]:
            candidates_k.append((lc, rc))

    recall = len(matches & set(candidates_k)) / len(matches)
    print(f"Recall@{K}: {recall:.4f}")

    return candidates_k


if __name__ == "__main__":
    for dataset, path in Datasets.items():
        dfs = [
            pd.read_csv(file, index_col="id") for file in sorted(path.glob("*_*.csv"))
        ]
        for df in dfs:
            df.fillna("", inplace=True)
        matches = set(
            pd.read_csv(path / "matches.csv").itertuples(index=False, name=None)
        )

        candidates_k = blocking(dataset, dfs, matches)
        candidates = pd.DataFrame(candidates_k, columns=["id_left", "id_right"])
        print(len(candidates))
        ldf = dfs[0]
        rdf = dfs[-1]
        ldf["record"] = ldf.astype(str).agg(" ".join, axis=1).str.strip()
        ldf = ldf[["record"]]
        rdf["record"] = rdf.astype(str).agg(" ".join, axis=1).str.strip()
        rdf = rdf[["record"]]

        # Sample records
        ldf = ldf.sample(n=200, random_state=42)

        ldf = pd.merge(
            candidates,
            ldf,
            left_on="id_left",
            right_index=True,
        )
        rdf = pd.merge(
            candidates,
            rdf,
            left_on="id_right",
            right_index=True,
        )
        candidates = pd.merge(
            ldf,
            rdf,
            on=["id_left", "id_right"],
            suffixes=("_left", "_right"),
        )
        candidates["label"] = candidates.apply(
            lambda row: (row["id_left"], row["id_right"]) in matches,
            axis=1,
        )
        print(candidates)
        candidates.to_csv(Path("data/llm4em") / f"{dataset}.csv", index=False)
