import math
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

# isort: split
from utils import APICostCalculator

cache = Cache("results/diskcache/matching_v2")
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


def score(
    instance,
    model,
    template=Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
"""
    ),
    use_prob: bool = False,
) -> list[float]:
    scores = []
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
            seed=42,
            temperature=0.0,
            logprobs=model.startswith("gpt"),
            top_logprobs=3 if model.startswith("gpt") else None,
            max_tokens=3,
        )
        if use_prob:
            assert model.startswith("gpt")
            content = response.choices[0].logprobs.content[0]
            if "yes" in content.token.strip().lower():
                scores.append(math.exp(content.logprob))
            elif "no" in content.token.strip().lower():
                scores.append(-math.exp(content.logprob))
            else:
                scores.append(0.0)
        else:
            content = response.choices[0].message.content.strip().lower()
            if "yes" in content:
                scores.append(1)
            elif "no" in content:
                scores.append(-1)
            else:
                scores.append(0)

    return scores


def pointwise_rank(instance, model: str = LLM) -> list[int]:
    scores = score(instance, model=model, use_prob=True)
    indexes = list(range(len(instance["candidates"])))
    indexes = [x for _, x in sorted(zip(scores, indexes), reverse=True)]
    return indexes


def match(instance, model: str = LLM, single_match: bool = False) -> list[bool]:
    scores = score(instance, model=model, use_prob=single_match)
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
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {api_cost_calculator.cost:.2f}")

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
