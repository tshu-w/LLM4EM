import math
import re
from collections.abc import Iterable, Iterator
from functools import partial, wraps
from pathlib import Path
from typing import Literal

import torch

torch.nn.CrossEntropyLoss = partial(torch.nn.CrossEntropyLoss, reduction="none")

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

cache = Cache("results/diskcache/c2f")
MODEL_DIR = Path("models/hf_models/")
MODEL_NAME = "flan-t5-xxl"
cache_hf = Cache(f"results/diskcache/{MODEL_NAME}")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR / MODEL_NAME)
MODEL = None
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR / MODEL_NAME, device_map="auto")
BATCH_SIZE = 32
TEMPLATE = Template(
    """Which of the following two records is more similar to the given record, i.e., there is no inconsistency in entity attributes? Answer only "Record A" or "Record B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
)

# From https://openai.com/pricing#language-models at 2024.01.01
MODEL_COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-1106": {"prompt": 0.0010, "completion": 0.0020},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
}
# Global variable to accumulate cost
ACCUMULATED_COST = 0


def chunks(iterable: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from iterable."""
    size = iterable.shape[0] if hasattr(iterable, "shape") else len(iterable)
    for i in range(0, size, n):
        yield iterable[i : i + n]


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


@cache_hf.memoize()
@torch.no_grad()
def cal_log_probs(
    sources: list[str],
    targets: list[str],
    tokenizer=TOKENIZER,
    model=MODEL,
) -> list[float]:
    inputs = tokenizer(
        text=sources,
        text_target=targets,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100
    outputs = model(**inputs, return_dict=True)
    log_probs = (-outputs.loss).view(inputs.labels.size(0), -1).mean(dim=1)
    return log_probs.tolist()


def compare(
    instance,
    template=Template(
        """Which of the following two records is more similar to the given record, i.e., there is no inconsistency in entity attributes? Answer only "Record A" or "Record B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
    ),
) -> bool:
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
    log_probs = cal_log_probs(sources, targets)
    probs = [math.exp(log_prob) for log_prob in log_probs]

    return probs[0] + probs[3] > probs[1] + probs[2]


def api_cost_decorator(model_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            cost = (
                MODEL_COST_PER_1K_TOKENS[model_name]["prompt"]
                * response.usage.prompt_tokens
                + MODEL_COST_PER_1K_TOKENS[model_name]["completion"]
                * response.usage.completion_tokens
            ) / 1000
            global ACCUMULATED_COST
            ACCUMULATED_COST += cost
            return response

        return wrapper

    return decorator


@api_cost_decorator(model_name="gpt-3.5-turbo")
@cache.memoize()
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    client=OpenAI(),
    model="gpt-3.5-turbo",
    **kwargs,
):
    return client.chat.completions.create(messages=messages, model=model, **kwargs)


def match(
    instance,
    model="gpt-3.5-turbo",
    template=Template(
        """Do the two entity records refer to the same real-world entity? Answer only "Yes" or "No".
Record 1: {{ record_left }}
Record 2: {{ record_right }}"""
    ),
) -> list[bool]:
    preds = []
    for candidate in instance["candidates"]:
        response = chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": template.render(
                        record_left=instance["anchor"],
                        record_right=candidate,
                    ),
                }
            ],
            model=model,
            logprobs=True,
            seed=42,
            temperature=0.0,
            max_tokens=5,
        )
        pred = response.choices[0].message.content.strip() not in [
            "No",
            "No.",
            "no",
            "no.",
        ]
        preds.append(pred)
    return preds


def select(
    instance,
    model="gpt-3.5-turbo",
    template=Template(
        """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer only with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
"""
    ),
) -> list[bool]:
    response = chat_complete(
        messages=[
            {
                "role": "user",
                "content": template.render(
                    anchor=instance["anchor"],
                    candidates=instance["candidates"],
                ),
            }
        ],
        model=model,
        logprobs=True,
        seed=42,
        temperature=0.0,
        max_tokens=5,
    )

    idx = re.search(r"([\d+])", response.choices[0].message.content.strip())
    preds = [False] * len(instance["candidates"])
    if idx:
        idx = int(idx.group(1))
        if 1 <= idx <= len(instance["candidates"]):
            preds[idx - 1] = True

    return preds


def coarse_to_fine(
    instance,
    mode: Literal["all", "knockout", "bubble"] = "all",
    topK=1,
) -> list[bool]:
    indexes = list(range(len(instance["candidates"])))
    n = len(indexes)

    if mode == "all":
        candidates = instance["candidates"]
        pairs = [
            (candidates[i], candidates[j]) for i in indexes for j in indexes if i != j
        ]
        sources = [
            TEMPLATE.render(anchor=instance["anchor"], cpair=p)
            for p in pairs
            for _ in range(2)
        ]
        targets = ["Record A", "Record B"] * len(pairs)
        pair_indexes = [k for i in indexes for j in indexes if i != j for k in (i, j)]

        log_probs = []
        for bs, bt in zip(chunks(sources, BATCH_SIZE), chunks(targets, BATCH_SIZE)):
            log_probs.extend(cal_log_probs(bs, bt))

        probs = [0] * len(indexes)
        for i, k in enumerate(pair_indexes):
            probs[k] += math.exp(log_probs[i])

        indexes = [x for _, x in sorted(zip(probs, indexes), reverse=True)]
    elif mode == "bubble":
        for i in range(topK):
            for j in range(n - i - 2, 0, -1):
                greater = compare(
                    instance={
                        "anchor": instance["anchor"],
                        "cpair": [
                            instance["candidates"][indexes[j + 1]],
                            instance["candidates"][indexes[j]],
                        ],
                    }
                )
                if greater:
                    indexes[j], indexes[j + 1] = indexes[j + 1], indexes[j]
    elif mode == "knockout":
        while len(indexes) > topK:
            winners = []
            for i in range(0, len(indexes), 2):
                if i + 1 < len(indexes):
                    greater = compare(
                        instance={
                            "anchor": instance["anchor"],
                            "cpair": [
                                instance["candidates"][indexes[i]],
                                instance["candidates"][indexes[i + 1]],
                            ],
                        }
                    )
                    if greater:
                        winners.append(indexes[i])
                    else:
                        winners.append(indexes[i + 1])
                else:
                    winners.append(indexes[i])

            indexes = winners

    preds = [False] * len(instance["candidates"])
    n_instance = {
        "anchor": instance["anchor"],
        "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
    }
    if topK == 1:
        n_preds = match(n_instance)
    else:
        n_preds = select(n_instance)

    for i, pred in enumerate(n_preds):
        preds[indexes[i]] = pred

    return preds


if __name__ == "__main__":
    ttl_preds = []
    ttl_labels = []
    ttl_cost = 0
    for file in Path("data/llm4em").glob("*.csv"):
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        groupby = list(
            df.groupby("record_left")[["record_right", "label"]]
            .apply(lambda x: x.to_dict("list"))
            .to_dict()
            .items()
        )
        instances = [
            {"anchor": k, "candidates": v["record_right"], "labels": v["label"]}
            for k, v in groupby
        ]

        preds_lst = thread_map(
            lambda it: coarse_to_fine(it, mode="all", topK=4),
            instances,
            max_workers=1,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {ACCUMULATED_COST:.2f}")

        ttl_preds.extend(preds)
        ttl_labels.extend(labels)
        ttl_cost += ACCUMULATED_COST
        ACCUMULATED_COST = 0

    print(classification_report(ttl_labels, ttl_preds, digits=4))
    print(
        f"Average Cost: {ttl_cost / len(list(Path('data/llm4em').glob('*.csv'))):.2f}"
    )
