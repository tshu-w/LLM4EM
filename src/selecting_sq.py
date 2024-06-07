import argparse
import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.utils import Seq2SeqWrapper


class SelectingSQ:
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
        model_name: str = "flan-t5-xxl",
        template: Template = template,
    ):
        self.wrapper = Seq2SeqWrapper(model_name)
        self.template = template

        cache = Cache(f"results/diskcache/selecting_{model_name}")
        self.wrapper.generate = cache.memoize(name="generate")(self.wrapper.generate)

    def __call__(self, instance) -> list[bool]:
        source = self.template.render(
            anchor=instance["anchor"],
            candidates=instance["candidates"],
        )
        target = self.wrapper.generate(source, max_new_tokens=32)

        idx = re.search(r"\[(\d+)\]", target.strip())
        preds = [False] * len(instance["candidates"])
        if idx:
            idx = int(idx.group(1))
            if 1 <= idx <= len(instance["candidates"]):
                preds[idx - 1] = True

        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="flan-t5-xxl", help="Name of the model to use"
    )
    args = parser.parse_args()

    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    selector = SelectingSQ(model_name=args.model)
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
            selector,
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
