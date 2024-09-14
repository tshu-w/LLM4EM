from pathlib import Path
from typing import Literal

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report
from tqdm.contrib.concurrent import thread_map

from src.matching import Matching
from src.utils import APICostCalculator, openai_chat_complete


class Comparing:
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
        model_name: str = "gpt-4o-mini",
        template: Template = template,
    ):
        self.model = model_name
        self.template = template

        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(f"results/diskcache/comparing_{model_name}")
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )
        self.matcher = Matching(model_name=model_name)

    def cmp(self, instance, compare_twice: bool = True) -> int:
        response1 = self.chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": self.template.render(
                        anchor=instance["anchor"],
                        cpair=instance["cpair"],
                    ),
                }
            ],
            model=self.model,
            seed=42,
            temperature=0.0,
            logprobs=self.model.startswith("gpt"),
            top_logprobs=3 if self.model.startswith("gpt") else None,
            max_tokens=3,
        )
        content1 = response1.choices[0].message.content.strip()
        score = 0
        if "A" in content1:
            score += 1
        elif "B" in content1:
            score -= 1

        if compare_twice:
            response2 = self.chat_complete(
                messages=[
                    {
                        "role": "user",
                        "content": self.template.render(
                            anchor=instance["anchor"],
                            cpair=instance["cpair"][::-1],
                        ),
                    }
                ],
                model=self.model,
                seed=42,
                temperature=0.0,
                logprobs=self.model.startswith("gpt"),
                top_logprobs=3 if self.model.startswith("gpt") else None,
                max_tokens=3,
            )
            content2 = response2.choices[0].message.content.strip()

            if "B" in content2:
                score += 1
            elif "A" in content2:
                score -= 1

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
        n_instance = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
        }
        n_preds = self.matcher(n_instance)

        for i, pred in enumerate(n_preds):
            preds[indexes[i]] = pred

        return preds

    @property
    def cost(self):
        return self.api_cost_decorator.cost + self.matcher.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = self.matcher.cost = value


if __name__ == "__main__":
    for pos in range(0, 10, 1):
        results = {}
        dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
        comparor = Comparing()
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
            instances = []
            for _, v in groupby:
                try:
                    idx = v["label"].index(True)
                    v["record_right"][pos], v["record_right"][idx] = (
                        v["record_right"][idx],
                        v["record_right"][pos],
                    )
                    v["label"][pos], v["label"][idx] = v["label"][idx], v["label"][pos]
                except ValueError:
                    pass

                instances.append(
                    {
                        "anchor": v["record_left"][0],
                        "candidates": v["record_right"],
                        "labels": v["label"],
                    }
                )

            preds_lst = thread_map(
                lambda it: comparor(it, mode="bubble"),  # noqa: B023
                instances,
                max_workers=16,
            )
            preds = [pred for preds in preds_lst for pred in preds]
            labels = [label for it in instances for label in it["labels"]]

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
