import re
from pathlib import Path
from typing import Literal

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

cache = Cache("results/diskcache/comparing")


@cache.memoize()
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    client=OpenAI(),
    model="gpt-3.5-turbo",
    **kwargs,
):
    return client.chat.completions.create(messages=messages, model=model, **kwargs)


def compare(
    instance,
    client=OpenAI(),
    model="gpt-3.5-turbo",
    template=Template(
        """Which of the following two records is more similar to the given record, i.e., there is no inconsistency in entity attributes? Answer only "A" or "B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
    ),
) -> bool | None:
    response = chat_complete(
        messages=[
            {
                "role": "user",
                "content": template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"],
                ),
            }
        ],
        model=model,
        logprobs=True,
        seed=42,
        temperature=0.0,
        max_tokens=5,
    )
    if "A" in response.choices[0].message.content.strip():
        return True
    elif "B" in response.choices[0].message.content.strip():
        return False
    else:
        return None


def match(
    instance,
    model="gpt-3.5-turbo",
    template=Template(
        """Do the two entity records refer to the same real-world entity? Answer only "Yes" or "No".
Record 1: {{ record_left }}
Record 2: {{ record_right }}"""
    ),
) -> bool:
    response = chat_complete(
        messages=[
            {
                "role": "user",
                "content": template.render(
                    record_left=instance["record_left"],
                    record_right=instance["record_right"],
                ),
            }
        ],
        model=model,
        logprobs=True,
        seed=42,
        temperature=0.0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip() not in ["No", "No.", "no", "no."]


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
    mode: Literal["bubble", "knockout"] = "bubble",
    topK=1,
) -> list[bool]:
    indexes = list(range(len(instance["candidates"])))
    n = len(indexes)

    if mode == "bubble":
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
    if topK == 1:
        m_instance = {
            "record_left": instance["anchor"],
            "record_right": instance["candidates"][indexes[0]],
        }
        preds[indexes[0]] = match(m_instance)
    else:
        s_instance = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
        }
        s_preds = select(s_instance)
        for i, pred in enumerate(s_preds):
            if pred:
                preds[indexes[i]] = True

    return preds


if __name__ == "__main__":
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
            lambda it: coarse_to_fine(it, mode="bubble", topK=1),
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds))
        print(confusion_matrix(labels[: len(preds)], preds))
