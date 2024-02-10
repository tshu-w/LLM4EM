import math
from functools import partial
from pathlib import Path
from typing import Literal

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
cache = Cache(f"results/diskcache/comparing_{MODEL_NAME}")


@cache.memoize(name="generate")
@torch.inference_mode()
def generate(
    source,
    **kwargs,
):
    input_ids = TOKENIZER(source, return_tensors="pt").input_ids.to("cuda")
    outputs = MODEL.generate(input_ids, **kwargs)
    target = TOKENIZER.decode(outputs.sequences[0], skip_special_tokens=True)
    return target


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


def cmp(
    instance,
    template=Template(
        """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer only "Record A" or "Record B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
    ),
    use_prob: bool = False,
) -> int:
    if not use_prob:
        source1 = template.render(
            anchor=instance["anchor"],
            cpair=instance["cpair"],
        )
        target1 = generate(
            source1,
            max_new_tokens=128,
            return_dict_in_generate=True,
        )
        source2 = template.render(
            anchor=instance["anchor"],
            cpair=instance["cpair"][::-1],
        )
        target2 = generate(
            source2,
            max_new_tokens=128,
            return_dict_in_generate=True,
        )
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
            template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"],
            ),
            template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"],
            ),
            template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"][::-1],
            ),
            template.render(
                anchor=instance["anchor"],
                cpair=instance["cpair"][::-1],
            ),
        ]
        targets = ["Record A", "Record B", "Record A", "Record B"]
        log_probs = []
        for bslice in gen_batches(len(sources), 2):
            log_probs.extend(cal_log_probs(sources[bslice], targets[bslice]))

        score = sum(math.exp(log_probs[i]) for i in [0, 4]) - sum(
            math.exp(log_probs[i]) for i in [1, 2]
        )

    return score


def pairwise_rank(
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
                    greater = cmp(
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

        indexes = [idx for _, idx in sorted(zip(scores, indexes), reverse=True)]
    elif mode == "bubble":
        for i in range(topK):
            for j in range(n - i - 1, 0, -1):
                greater = cmp(
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
                    greater = cmp(
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


def compare(
    instance,
    mode: Literal["all", "bubble", "knockout"] = "bubble",
    topK: int = 1,
) -> list[bool]:
    indexes = pairwise_rank(instance, mode=mode, topK=topK)
    preds = [False] * len(instance["candidates"])
    # n_instance = {
    #     "anchor": instance["anchor"],
    #     "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
    # }
    # n_preds = matching_v2.match(n_instance)

    # for i, pred in enumerate(n_preds):
    #     preds[indexes[i]] = pred

    for idx in indexes[:topK]:
        preds[idx] = True
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
            lambda it: compare(it, mode="bubble"),
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
