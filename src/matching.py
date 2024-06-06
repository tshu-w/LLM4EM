import math
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.utils import APICostCalculator, openai_chat_complete


class Matching:
    template = Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
"""
    )

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0613",
        template: Template = template,
    ):
        self.model = model_name
        self.template = template

        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(f"results/diskcache/matching_{model_name}")
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )

    def score(self, instance, use_prob: bool = False) -> list[float]:
        scores = []
        for candidate in instance["candidates"]:
            response = self.chat_complete(
                messages=[
                    {
                        "role": "user",
                        "content": self.template.render(
                            record_left=instance["anchor"],
                            record_right=candidate,
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
            if use_prob:
                assert self.model.startswith("gpt")
                content = response.choices[0].logprobs.content[0]
                if "yes" in content.token.strip().lower():
                    scores.append(math.exp(content.logprob))
                elif "no" in content.token.strip().lower():
                    scores.append(-math.exp(content.logprob))
                else:
                    scores.append(0.0)
            else:
                content = response.choices[0].message.content.strip().lower()
                if "yes" in content:
                    scores.append(1)
                elif "no" in content:
                    scores.append(-1)
                else:
                    scores.append(0)

        return scores

    def pointwise_rank(self, instance) -> list[int]:
        scores = self.score(instance, use_prob=True)
        indexes = list(range(len(instance["candidates"])))
        indexes = [
            x for _, x in sorted(zip(scores, indexes, strict=True), reverse=True)
        ]
        return indexes

    def __call__(
        self,
        instance,
        single_match: bool = False,
    ) -> list[bool]:
        scores = self.score(instance)
        if single_match:
            max_score = max(scores)
            preds = [sc >= max_score and sc > 0 for sc in scores]
        else:
            preds = [sc > 0 for sc in scores]

        return preds

    @property
    def cost(self):
        return self.api_cost_decorator.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    matcher = Matching()
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
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {matcher.cost:.2f}")

        results[dataset] = classification_report(
            labels[: len(preds)], preds, output_dict=True
        )["True"]
        results[dataset].pop("support")
        for k, v in results[dataset].items():
            results[dataset][k] = v * 100
        results[dataset]["cost"] = matcher.cost
        matcher.cost = 0

    results["mean"] = {
        "precision": sum(v["precision"] for v in results.values()) / len(results),
        "recall": sum(v["recall"] for v in results.values()) / len(results),
        "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
        "cost": sum(v["cost"] for v in results.values()) / len(results),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    print(df)
    print(df.to_csv(float_format="%.2f", index=False))
