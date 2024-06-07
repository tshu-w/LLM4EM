import argparse
import math
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import gen_batches
from tqdm.contrib.concurrent import thread_map

from src.utils import Seq2SeqWrapper


class MatchingSQ:
    template = Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
"""
    )

    def __init__(
        self,
        model_name: str = "flan-t5-xxl",
        template: Template = template,
    ):
        self.wrapper = Seq2SeqWrapper(model_name)
        self.template = template

        cache = Cache(f"results/diskcache/matching_{model_name}")
        self.wrapper.generate = cache.memoize(name="generate")(self.wrapper.generate)
        self.wrapper.cal_log_probs = cache.memoize(name="cal_log_probs")(
            self.wrapper.cal_log_probs
        )

    def score(self, instance, use_prob: bool = False) -> list[float]:
        if not use_prob:
            scores = []
            for candidate in instance["candidates"]:
                source = self.template.render(
                    record_left=instance["anchor"],
                    record_right=candidate,
                )
                target = (
                    self.wrapper.generate(source, max_new_tokens=32).strip().lower()
                )
                if "yes" in target:
                    scores.append(1)
                elif "no" in target:
                    scores.append(-1)
                else:
                    scores.append(0)

            return scores
        else:
            sources = [
                self.template.render(
                    record_left=instance["anchor"],
                    record_right=candidate,
                )
                for candidate in instance["candidates"]
                for _ in range(2)
            ]
            targets = ["Yes", "No"] * len(instance["candidates"])
            log_probs = []
            for bslice in gen_batches(len(sources), 2):
                log_probs.extend(
                    self.wrapper.cal_log_probs(sources[bslice], targets[bslice])
                )
            probs = [0] * len(instance["candidates"])
            for i in range(len(instance["candidates"])):
                if log_probs[i * 2] >= log_probs[i * 2 + 1]:
                    probs[i] = math.exp(log_probs[i * 2])
                else:
                    probs[i] = -math.exp(log_probs[i * 2 + 1])

            return probs

    def pointwise_rank(self, instance) -> list[int]:
        scores = self.score(instance, use_prob=True)
        indexes = list(range(len(instance["candidates"])))
        indexes = [
            x for _, x in sorted(zip(scores, indexes, strict=True), reverse=True)
        ]
        return indexes

    def __call__(self, instance, single_match: bool = False) -> list[bool]:
        scores = self.score(instance, use_prob=single_match)
        if single_match:
            max_score = max(scores)
            preds = [sc >= max_score and sc > 0 for sc in scores]
        else:
            preds = [sc > 0 for sc in scores]

        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="flan-t5-xxl", help="Name of the model to use"
    )
    args = parser.parse_args()

    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    matcher = MatchingSQ(model_name=args.model)
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
            matcher,
            instances,
            max_workers=1,
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

    results["mean"] = {
        "precision": sum(v["precision"] for v in results.values()) / len(results),
        "recall": sum(v["recall"] for v in results.values()) / len(results),
        "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    print(df)
    print(df.to_csv(float_format="%.2f", index=False))
