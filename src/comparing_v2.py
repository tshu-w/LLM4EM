import math
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

# isort: split
import matching_v2
from utils import APICostCalculator

cache = Cache("results/diskcache/comparing_v2")
LLM = "gpt-3.5-turbo-0613"
api_cost_calculator = APICostCalculator(model_name=LLM)


@api_cost_calculator
@cache.memoize(name="chat_complete")
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    model,
    client=OpenAI(),
    **kwargs,
):
    response = client.chat.completions.create(messages=messages, model=model, **kwargs)
    if response.choices is None:
        raise ValueError(f"Error response: {response}")
    return response


def cmp(
    instance,
    model,
    template=Template(
        """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer with the corresponding record identifier "A" or "B".

Given entity record:
{{ anchor }}

Record A: {{ cpair[0] }}
Record B: {{ cpair[1] }}
"""
    ),
) -> float:
    response1 = chat_complete(
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
        seed=42,
        temperature=0.0,
        logprobs=True,
        top_logprobs=3,
        max_tokens=3,
    )
    response2 = chat_complete(
        messages=[
            {
                "role": "user",
                "content": template.render(
                    anchor=instance["anchor"],
                    cpair=instance["cpair"][::-1],
                ),
            }
        ],
        model=model,
        seed=42,
        temperature=0.0,
        logprobs=True,
        top_logprobs=3,
        max_tokens=3,
    )
    prob = 0
    if "A" in response1.choices[0].message.content.strip():
        prob += math.exp(response1.choices[0].logprobs.content[0].logprob)
    elif "B" in response1.choices[0].message.content.strip():
        prob -= math.exp(response1.choices[0].logprobs.content[0].logprob)

    if "B" in response2.choices[0].message.content.strip():
        prob += math.exp(response1.choices[0].logprobs.content[0].logprob)
    elif "A" in response2.choices[0].message.content.strip():
        prob -= math.exp(response1.choices[0].logprobs.content[0].logprob)

    return prob


def pairwise_rank(
    instance,
    model: str = LLM,
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
                        model=model,
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
                    model=model,
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
                        model=model,
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
    model: str = LLM,
    mode: Literal["all", "bubble", "knockout"] = "bubble",
    topK: int = 1,
) -> list[bool]:
    indexes = pairwise_rank(instance, model=model, mode=mode)
    preds = [False] * len(instance["candidates"])
    n_instance = {
        "anchor": instance["anchor"],
        "candidates": [instance["candidates"][idx] for idx in indexes[:topK]],
    }
    n_preds = matching_v2.match(n_instance, model=LLM)

    for i, pred in enumerate(n_preds):
        preds[indexes[i]] = pred

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
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {api_cost_calculator.cost:.2f}")
        print(f"Matching Cost: {matching_v2.api_cost_calculator.cost:.2f}")

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
    print(f"Cost: {api_cost_calculator.cost:.2f}")
    print(f"Matching Cost: {matching_v2.api_cost_calculator.cost:.2f}")
