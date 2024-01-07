import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

cache = Cache("results/diskcache/selecting")


@cache.memoize()
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    client=OpenAI(),
    model="gpt-3.5-turbo",
    **kwargs,
):
    return client.chat.completions.create(messages=messages, model=model, **kwargs)


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
        idx = max(idx, 1)
        idx = min(idx, len(instance["candidates"]))
        preds[idx - 1] = True

    return preds


if __name__ == "__main__":
    for file in Path("data/llm4em").glob("*.csv"):
        dataset = file.stem
        if dataset == "dblp-scholar":
            continue
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)
        from sklearn.utils import shuffle

        df = shuffle(df, random_state=42)

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
            select,
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds))
        print(confusion_matrix(labels[: len(preds)], preds))
