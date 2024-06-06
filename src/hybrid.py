from collections import deque
from pathlib import Path
from typing import Literal

import pandas as pd
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.comparing import Comparing
from src.matching import Matching
from src.selecting import Selecting


class Hybrid:
    ranking_strategy: Literal["matching", "comparing"] = "matching"

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0613",
        ranking_strategy: Literal["matching", "comparing"] = ranking_strategy,
    ):
        self.model = model_name
        self.ranking_strategy = ranking_strategy
        if self.ranking_strategy == "matching":
            self.ranker = Matching(model_name=model_name)
        elif self.ranking_strategy == "comparing":
            self.ranker = Comparing(model_name=model_name)

        self.selector = Selecting(model_name=model_name)

    def __call__(self, instance, topK: int = 1) -> list[bool]:
        if self.ranking_strategy == "matching":
            indexes = self.ranker.pointwise_rank(instance)
        elif self.ranking_strategy == "comparing":
            indexes = self.ranker.pairwise_rank(instance, topK=topK)

        indexes_k = indexes[:topK]
        preds = [False] * len(instance["candidates"])
        dq = deque(indexes[:topK])
        dq.rotate(2)
        indexes_k = list(dq)
        instance_k = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][idx] for idx in indexes_k],
        }
        preds_k = self.selector(instance_k)
        for i, pred in enumerate(preds_k):
            preds[indexes_k[i]] = pred

        return preds

    @property
    def cost(self):
        return self.ranker.cost + self.selector.cost

    @cost.setter
    def cost(self, value: int):
        self.ranker.cost = self.selector.cost = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    hybrid = Hybrid()
    for file in dataset_files:
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        groupby = list(
            df.groupby("id_left")[["record_left", "record_right", "label"]]
            .apply(lambda x: x.to_dict("list"))
            .to_dict()
            .items()
        )
        instances = [
            {
                "anchor": v["record_left"][0],
                "candidates": v["record_right"],
                "labels": v["label"],
            }
            for _, v in groupby
        ]

        preds_lst = thread_map(
            lambda it: hybrid(it, topK=4),
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))

        results[dataset] = classification_report(
            labels[: len(preds)], preds, output_dict=True
        )["True"]
        results[dataset].pop("support")
        for k, v in results[dataset].items():
            results[dataset][k] = v * 100
        results[dataset]["cost"] = hybrid.cost
        hybrid.cost = 0

    results["mean"] = {
        "precision": sum(v["precision"] for v in results.values()) / len(results),
        "recall": sum(v["recall"] for v in results.values()) / len(results),
        "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
        "cost": sum(v["cost"] for v in results.values()) / len(results),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    print(df)
    print(df.to_csv(float_format="%.2f", index=False))
