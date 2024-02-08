import math
from functools import partial
from pathlib import Path

import torch

torch.nn.CrossEntropyLoss = partial(torch.nn.CrossEntropyLoss, reduction="none")

import pandas as pd
from diskcache import Cache
from fastchat.model import get_conversation_template, load_model
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import gen_batches
from tqdm.contrib.concurrent import thread_map

MODEL_DIR = Path("models/hf_models/")
MODEL_NAME = "Mixtral-7B-Instruct-v0.1"
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR / MODEL_NAME)
# MODEL = None
MODEL, TOKENIZER = load_model(
    MODEL_DIR / MODEL_NAME,
    device="cuda",
    dtype=torch.float16,
    num_gpus=1,
)
cache = Cache(f"results/diskcache/matching_{MODEL_NAME}")
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token


@cache.memoize(name="generate")
@torch.inference_mode()
def generate(
    source,
    **kwargs,
):
    conv = get_conversation_template(MODEL_NAME)
    conv.append_message(conv.roles[0], source)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = TOKENIZER(prompt, return_tensors="pt").to("cuda")
    outputs = MODEL.generate(**inputs, **kwargs, pad_token_id=TOKENIZER.pad_token_id)
    target = TOKENIZER.decode(outputs.sequences[0], skip_special_tokens=True)
    return target[len(prompt) :]


@cache.memoize(name="cal_log_probs")
@torch.inference_mode()
def cal_log_probs(
    sources: list[str],
    targets: list[str],
) -> list[float]:
    psources = []
    ptargets = []
    for src, tgt in zip(sources, targets):
        conv = get_conversation_template(MODEL_NAME)
        conv.append_message(conv.roles[0], src)
        conv.append_message(conv.roles[1], None)
        psources.append(conv.get_prompt())
        conv = get_conversation_template(MODEL_NAME)
        conv.append_message(conv.roles[0], src)
        conv.append_message(conv.roles[1], tgt)
        ptargets.append(conv.get_prompt())

    inputs = TOKENIZER(
        text=psources,
        text_target=ptargets,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    inputs.pop("token_type_ids", None)

    source_lengths = torch.tensor([len(src) for src in inputs["input_ids"]]).unsqueeze(
        -1
    )
    range_tensor = torch.arange(inputs["labels"].size(1)).expand_as(inputs["labels"])
    mask = range_tensor < source_lengths
    inputs["input_ids"] = inputs["labels"].clone()
    inputs["labels"][mask] = -100

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
    use_prob: bool = False,
) -> list[float]:
    if not use_prob:
        scores = []
        for candidate in instance["candidates"]:
            source = template.render(
                record_left=instance["anchor"],
                record_right=candidate,
            )
            target = (
                generate(
                    source,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                )
                .strip()
                .lower()
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
            template.render(
                record_left=instance["anchor"],
                record_right=candidate,
            )
            for candidate in instance["candidates"]
            for _ in range(2)
        ]
        targets = ["Yes", "No"] * len(instance["candidates"])
        log_probs = []
        for bslice in gen_batches(len(sources), 2):
            log_probs.extend(cal_log_probs(sources[bslice], targets[bslice]))
        probs = [0] * len(instance["candidates"])
        for i in range(len(instance["candidates"])):
            if log_probs[i * 2] >= log_probs[i * 2 + 1]:
                probs[i] = math.exp(log_probs[i * 2])
            else:
                probs[i] = -math.exp(log_probs[i * 2 + 1])

        return probs


def pointwise_rank(instance) -> list[int]:
    scores = score(instance, use_prob=True)
    indexes = list(range(len(instance["candidates"])))
    indexes = [x for _, x in sorted(zip(scores, indexes), reverse=True)]
    return indexes


def match(instance, single_match: bool = False) -> list[bool]:
    scores = score(instance, use_prob=single_match)
    if single_match:
        max_score = max(scores)
        preds = [sc >= max_score and sc > 0 for sc in scores]
    else:
        preds = [sc > 0 for sc in scores]

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
