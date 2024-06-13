import argparse
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import gen_batches
from tqdm.contrib.concurrent import thread_map

from src.matching_sq import MatchingSQ
from src.utils import Seq2SeqWrapper


class ComparingSQ:
    template = Template(
        """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer with the corresponding record identifier "Record A" or "Record B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
    )

    def __init__(
        self,
        model_name: str = "flan-t5-xxl",
        template: Template = template,
    ):
        self.wrapper = Seq2SeqWrapper(model_name)
        self.template = template
        self.matcher = MatchingSQ(model_name)

        cache = Cache(f"results/diskcache/comparing_{model_name}")
        self.wrapper.generate = cache.memoize(name="generate")(self.wrapper.generate)
        self.wrapper.cal_log_probs = cache.memoize(name="cal_log_probs")(
            self.wrapper.cal_log_probs
        )

    def cmp(self, instance, use_prob: bool = False) -> int:
        if not use_prob:
            source1 = self.template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"],
            )
            target1 = self.wrapper.generate(source1, max_new_tokens=32)
            source2 = self.template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"][::-1],
            )
            target2 = self.wrapper.generate(source2, max_new_tokens=32)
            score = 0
            if "Record A" in target1:
                score += 1
            elif "Record B" in target1:
                score -= 1

            if "Record B" in target2:
                score += 1
            elif "Record A" in target2:
                score -= 1
        else:
            sources = [
                self.template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"],
                ),
                self.template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"],
                ),
                self.template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"][::-1],
                ),
                self.template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"][::-1],
                ),
            ]
            targets = ["Record A", "Record B", "Record A", "Record B"]
            log_probs = []
            for bslice in gen_batches(len(sources), 2):
                log_probs.extend(
                    self.wrapper.cal_log_probs(sources[bslice], targets[bslice])
                )

            score = sum(math.exp(log_probs[i]) for i in [0, 3]) - sum(
                math.exp(log_probs[i]) for i in [1, 2]
            )

        return score

    def pairwise_rank(
        self,
        instance,
        mode: Literal["all", "bubble", "knockout"] = "bubble",
        topK: int = 1,
    ) -> list[int]:
        indexes = list(range(len(instance["candidates"])))
        n = len(indexes)

        if mode == "all":
            scores = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        greater = self.cmp(
                            instance={
                                "anchor": instance["anchor"],
                                "cpair": [
                                    instance["candidates"][i],
                                    instance["candidates"][j],
                                ],
                            },
                        )
                        if greater > 0:
                            scores[i] += greater
                        elif greater < 0:
                            scores[j] += -greater

            indexes = [
                idx
                for _, idx in sorted(zip(scores, indexes, strict=True), reverse=True)
            ]
        elif mode == "bubble":
            for i in range(topK):
                for j in range(n - i - 1, 0, -1):
                    greater = self.cmp(
                        instance={
                            "anchor": instance["anchor"],
                            "cpair": [
                                instance["candidates"][indexes[j]],
                                instance["candidates"][indexes[j - 1]],
                            ],
                        },
                    )
                    if greater >= 0:
                        indexes[j], indexes[j - 1] = indexes[j - 1], indexes[j]
        elif mode == "knockout":
            while len(indexes) > topK:
                winners = []
                for i in range(0, len(indexes), 2):
                    if i + 1 < len(indexes):
                        greater = self.cmp(
                            instance={
                                "anchor": instance["anchor"],
                                "cpair": [
                                    instance["candidates"][indexes[i]],
                                    instance["candidates"][indexes[i + 1]],
                                ],
                            },
                        )
                        if greater >= 0:
                            winners.append(indexes[i])
                        else:
                            winners.append(indexes[i + 1])
                    else:
                        winners.append(indexes[i])

                indexes = winners

        return indexes

    def __call__(
        self,
        instance,
        mode: Literal["all", "bubble", "knockout"] = "bubble",
        topK: int = 1,
    ) -> list[bool]:
        indexes = self.pairwise_rank(instance, mode=mode, topK=topK)
        preds = [False] * len(instance["candidates"])

        # for idx in indexes[:topK]:
        #     preds[idx] = True
        # return preds
        n_instance = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
        }
        n_preds = self.matcher(n_instance)

        for i, pred in enumerate(n_preds):
            preds[indexes[i]] = pred

        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="flan-t5-xxl", help="Name of the model to use"
    )
    args = parser.parse_args()

    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    comparor = ComparingSQ(model_name=args.model)
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
            lambda it: comparor(it, mode="bubble"),
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
