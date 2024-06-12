import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report
from tqdm.contrib.concurrent import thread_map

from src.utils import APICostCalculator, openai_chat_complete


class Selecting:
    template = Template(
        """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
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
        cache = Cache(f"results/diskcache/selecting_{model_name}")
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )

    def __call__(self, instance) -> list[bool]:
        response = self.chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": self.template.render(
                        anchor=instance["anchor"],
                        candidates=instance["candidates"],
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

        idx = re.search(r"\[(\d+)\]", response.choices[0].message.content.strip())
        preds = [False] * len(instance["candidates"])
        if idx:
            idx = int(idx.group(1))
            if 1 <= idx <= len(instance["candidates"]):
                preds[idx - 1] = True

        return preds

    @property
    def cost(self):
        return self.api_cost_decorator.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = value


if __name__ == "__main__":
    for pos in range(0, 10, 2):
        results = {}
        dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
        selector = Selecting()
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
                selector,
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
