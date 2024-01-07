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

Given an entity record:
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
        idx = max(idx, 1)
        idx = min(idx, len(instance["candidates"]))
        preds[idx - 1] = True

    return preds


def knockout(
    instance,
    top_k=1,
) -> int:
    """
    A single elimination tournament where the loser of each match is immediately eliminated from the tournament.
    """
    participants = list(range(len(instance["candidates"])))

    while len(participants) > top_k:
        winners = []
        for i in range(0, len(participants), 2):
            if i + 1 < len(participants):
                res = compare(
                    instance={
                        "anchor": instance["anchor"],
                        "cpair": [
                            instance["candidates"][participants[i]],
                            instance["candidates"][participants[i + 1]],
                        ],
                    }
                )
                if res is not None:
                    if res is True:
                        winners.append(participants[i])
                    elif res is False:
                        winners.append(participants[i + 1])
            else:
                winners.append(participants[i])

        participants = winners

    preds = [False] * len(instance["candidates"])
    if top_k == 1:
        winner = participants[0]
        m_instance = {
            "record_left": instance["anchor"],
            "record_right": instance["candidates"][winner],
        }
        preds[winner] = match(m_instance)
    else:
        winners = participants
        s_instance = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][winner] for winner in winners],
        }
        s_preds = select(s_instance)
        for i, pred in enumerate(s_preds):
            if pred:
                preds[winners[i]] = True

    return preds


if __name__ == "__main__":
    for file in Path("data/llm4em").glob("*.csv"):
        dataset = file.stem
        if dataset == "dblp-scholar":
            continue
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
            knockout,
            instances[:2],
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds))
        print(confusion_matrix(labels[: len(preds)], preds))
