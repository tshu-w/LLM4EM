# ruff: noqa: B023
from pathlib import Path

import nltk
import pandas as pd
from retriv import SparseRetriever
from sklearn.utils import shuffle

Datasets = {
    "abt-buy": (Path("data/pyJedAI/data/ccer/D2"), ("abt.csv", "buy.csv"), "|"),
    "amazon-google": (Path("data/pyJedAI/data/ccer/D3"), ("amazon.csv", "gp.csv"), "#"),
    "dblp-acm": (Path("data/pyJedAI/data/ccer/D4"), ("dblp.csv", "acm.csv"), "%"),
    "dblp-scholar": (
        Path("data/pyJedAI/data/ccer/D9"),
        ("dblp.csv", "scholar.csv"),
        ">",
    ),
    "imdb-tmdb": (Path("data/pyJedAI/data/ccer/D5"), ("imdb.csv", "tmdb.csv"), "|"),
    "imdb-tvdb": (Path("data/pyJedAI/data/ccer/D6"), ("imdb.csv", "tvdb.csv"), "|"),
    "tmdb-tvdb": (Path("data/pyJedAI/data/ccer/D7"), ("tmdb.csv", "tvdb.csv"), "|"),
    "walmart-amazon": (
        Path("data/pyJedAI/data/ccer/D8"),
        ("walmart.csv", "amazon.csv"),
        "|",
    ),
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
    topK: int = 20,
):
    try:
        retriever = SparseRetriever.load(f"{dataset}-index")
        # raise FileNotFoundError
    except FileNotFoundError:
        retriever = SparseRetriever(index_name=f"{dataset}-index")
        retriever = retriever.index(
            generate_docs(dfs[-1]),
            show_progress=True,
        )

    queries = list(generate_docs(dfs[0]))
    candidates = retriever.bsearch(queries, show_progress=True, cutoff=topK)
    candidates_k = list()
    for lc, v in candidates.items():
        for rc in sorted(v, key=v.get, reverse=True):
            candidates_k.append((lc, rc))

    recall = len(matches & set(candidates_k)) / len(matches) * 100
    print(f"Recall@{topK}: {recall:.2f}")

    return candidates_k


if __name__ == "__main__":
    for dataset, (path, files, sep) in Datasets.items():
        print(dataset)
        dfs = [pd.read_csv(path / f, index_col="id", sep=sep, dtype=str) for f in files]
        for df in dfs:
            df.fillna("", inplace=True)
            df.drop(columns=["description"], inplace=True, errors="ignore")
            df.drop(
                columns=["http://dbpedia.org/ontology/abstract"],
                inplace=True,
                errors="ignore",
            )

        matches = pd.read_csv(path / "gt.csv", sep=sep)
        matches.columns = ["id_left", "id_right"]
        match_set = set(matches.itertuples(index=False, name=None))

        # # HACK: We foget to update the retrieval index with code update, so we build the index with old data
        # old_dfs = [pd.read_csv(path / f, index_col="id", sep=sep) for f in files]
        # for df in old_dfs:
        #     df.fillna("", inplace=True)

        # _ = blocking(dataset, old_dfs, match_set)

        candidates = blocking(dataset, dfs, match_set)
        candidates = pd.DataFrame(candidates, columns=["id_left", "id_right"])

        ldf = dfs[0]
        rdf = dfs[-1]
        ldf["record"] = ldf.astype(str).agg(" ".join, axis=1).str.strip()
        ldf = ldf[["record"]]
        rdf["record"] = rdf.astype(str).agg(" ".join, axis=1).str.strip()
        rdf = rdf[["record"]]

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
            lambda row: (row["id_left"], row["id_right"]) in match_set,
            axis=1,
        )

        # Sample 400 records, 300 with matches, 100 without matches
        topK = 10
        candidates_K = (
            candidates.groupby("id_left")
            .filter(lambda x: len(x) > topK)
            .groupby("id_left")
            .head(topK)
        )
        wm = candidates_K[candidates_K["label"]].sample(n=300, random_state=42)[
            "id_left"
        ]
        wom = (
            candidates_K[~candidates_K["id_left"].isin(wm)]["id_left"]
            .drop_duplicates()
            .sample(n=100, random_state=42)
        )

        to_remove = set(
            matches[matches["id_left"].isin(wom)].itertuples(index=False, name=None)
        )
        candidates_sample = candidates[
            ~candidates.apply(
                lambda row: (row["id_left"], row["id_right"]) in to_remove, axis=1
            )
        ]

        candidates_sample = candidates_sample[
            candidates_sample["id_left"].isin(wm)
            | candidates_sample["id_left"].isin(wom)
        ]
        candidates_sample = candidates_sample.groupby("id_left").head(topK)
        candidates_remains = candidates[~candidates.index.isin(candidates_sample.index)]
        assert len(candidates_sample) + len(candidates_remains) == len(candidates)

        candidates_sample = shuffle(
            candidates_sample.reset_index(drop=True), random_state=42
        )
        candidates_sample.to_csv(Path("data/llm4em") / f"{dataset}.csv", index=False)

        # Prepare training data for Baselines
        candidates_remains = shuffle(
            candidates_remains.reset_index(drop=True), random_state=42
        )
        for key in ["record_left", "record_right"]:
            candidates_sample[key] = candidates_sample[key].str.replace("\t", " ")
            candidates_remains[key] = candidates_remains[key].str.replace("\t", " ")

        candidates_remains["label"] = candidates_remains["label"].astype(int)
        candidates_sample["label"] = candidates_sample["label"].astype(int)

        # Sudowoodo
        Path(f"data/llm4em/sudowoodo/{dataset}").mkdir(parents=True, exist_ok=True)
        candidates_remains[["record_left", "record_right", "label"]][:5000].to_csv(
            Path(f"data/llm4em/sudowoodo/{dataset}") / "train.txt",
            index=False,
            sep="\t",
            header=False,
        )
        candidates_sample[["record_left", "record_right", "label"]].to_csv(
            Path(f"data/llm4em/sudowoodo/{dataset}") / "test.txt",
            index=False,
            sep="\t",
            header=False,
        )

        allpair = pd.concat([candidates_remains[:5000], candidates_sample])
        allpair[["record_left", "record_right"]][: topK * len(dfs[0])].to_csv(
            Path(f"data/llm4em/sudowoodo/{dataset}") / "train_no_label.txt",
            index=False,
            sep="\t",
            header=False,
        )

        # Ditto, HierGAT
        train_df = pd.concat(
            [
                candidates_remains[candidates_remains["label"] == 1][:500],
                candidates_remains[candidates_remains["label"] == 0][:4500],
            ]
        )[["record_left", "record_right", "label"]]
        Path(f"data/llm4em/sft/{dataset}").mkdir(parents=True, exist_ok=True)
        from sklearn.model_selection import train_test_split

        train, valid = train_test_split(train_df, test_size=0.2)
        train.to_csv(
            Path(f"data/llm4em/sft/{dataset}") / "train.txt",
            index=False,
            sep="\t",
            header=False,
        )
        valid.to_csv(
            Path(f"data/llm4em/sft/{dataset}") / "valid.txt",
            index=False,
            sep="\t",
            header=False,
        )
        candidates_sample[["record_left", "record_right", "label"]].to_csv(
            Path(f"data/llm4em/sft/{dataset}") / "test.txt",
            index=False,
            sep="\t",
            header=False,
        )
