import re
from pathlib import Path

import pandas as pd
import torch
from diskcache import Cache
from fastchat.model import get_conversation_template, load_model
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

MODEL_DIR = Path("models/hf_models/")
MODEL_NAME = "Mistral-7B-Instruct-v0.1"
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR / MODEL_NAME)
# MODEL = None
MODEL, TOKENIZER = load_model(
    MODEL_DIR / MODEL_NAME,
    device="cuda",
    dtype=torch.float16,
    num_gpus=1,
)
cache = Cache(f"results/diskcache/selecting_{MODEL_NAME}")
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


def select(
    instance,
    template=Template(
        """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
"""
    ),
) -> list[bool]:
    source = template.render(
        anchor=instance["anchor"],
        candidates=instance["candidates"],
    )
    target = generate(
        source,
        max_new_tokens=128,
        return_dict_in_generate=True,
    )

    idx = re.search(r"\[(\d+)\]", target.strip())
    preds = [False] * len(instance["candidates"])
    if idx:
        idx = int(idx.group(1))
        if 1 <= idx <= len(instance["candidates"]):
            preds[idx - 1] = True

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
            select,
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