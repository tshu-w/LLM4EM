import math
from functools import partial
from pathlib import Path

import torch

torch.nn.CrossEntropyLoss = partial(torch.nn.CrossEntropyLoss, reduction="none")

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import gen_batches
from tqdm.contrib.concurrent import thread_map
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = Path("models/hf_models/")
MODEL_NAME = "flan-t5-xxl"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR / MODEL_NAME)
MODEL = None
MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR / MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)
BATCH_SIZE = 2
cache = Cache(f"results/diskcache/matching_{MODEL_NAME}")


@cache.memoize(name="cal_log_probs")
@torch.inference_mode()
def cal_log_probs(
    sources: list[str],
    targets: list[str],
) -> list[float]:
    inputs = TOKENIZER(
        text=sources,
        text_target=targets,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    inputs["labels"][inputs["labels"] == TOKENIZER.pad_token_id] = -100
    outputs = MODEL(**inputs, return_dict=True)
    log_probs = (-outputs.loss).view(inputs.labels.size(0), -1).mean(dim=1)
    return log_probs.tolist()


def score(
    instance,
    template=Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
"""
    ),
) -> list[float]:
    sources = [
        template.render(
            record_left=instance["anchor"],
            record_right=candidate,
        )
        for candidate in instance["candidates"]
        for _ in range(2)
    ]
    targets = ["Yes", "No"] * len(instance["candidates"])
    log_probs = []
    for bslice in gen_batches(len(sources), BATCH_SIZE):
        log_probs.extend(cal_log_probs(sources[bslice], targets[bslice]))
    probs = [0] * len(instance["candidates"])
    for i in range(len(instance["candidates"])):
        if log_probs[i * 2] >= log_probs[i * 2 + 1]:
            probs[i] = math.exp(log_probs[i * 2])
        else:
            probs[i] = -math.exp(log_probs[i * 2 + 1])

    return probs


def pointwise_rank(instance) -> list[int]:
    probs = score(instance)
    indexes = list(range(len(instance["candidates"])))
    indexes = [x for _, x in sorted(zip(probs, indexes), reverse=True)]
    return indexes


def match(instance, single_match: bool = False) -> list[bool]:
    probs = score(instance)
    if single_match:
        max_prob = max(probs)
        preds = [prob >= max_prob and prob > 0 for prob in probs]
    else:
        preds = [prob > 0 for prob in probs]

    return preds


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
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
            match,
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
