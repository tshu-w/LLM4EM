import re
from functools import wraps
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

cache = Cache("results/diskcache/selecting")

# From https://openai.com/pricing#language-models at 2024.01.01
MODEL_COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-3.5-turbo-1106": {"prompt": 0.0010, "completion": 0.0020},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
}
LLM = "gpt-3.5-turbo"
# Global variable to accumulate cost
ACCUMULATED_COST = 0


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


@api_cost_decorator(model_name=LLM)
@cache.memoize(name="__main__.chat_complete")
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def chat_complete(
    messages,
    model=LLM,
    client=OpenAI(),
    **kwargs,
):
    return client.chat.completions.create(messages=messages, model=model, **kwargs)


def select(
    instance,
    model=LLM,
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

    idx = re.search(r"\[(\d+)\]", response.choices[0].message.content.strip())
    preds = [False] * len(instance["candidates"])
    if idx:
        idx = int(idx.group(1))
        if 1 <= idx <= len(instance["candidates"]):
            preds[idx - 1] = True

    return preds


if __name__ == "__main__":
    for pos in range(0, 10):
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
            instances = []
            for _, v in groupby:
                try:
                    idx = v["label"].index(True)
                    v["record_right"][pos], v["record_right"][idx] = (
                        v["record_right"][idx],
                        v["record_right"][pos],
                    )
                    v["label"][pos], v["label"][idx] = v["label"][idx], v["label"][pos]
                except ValueError:
                    pass

                instances.append(
                    {
                        "anchor": v["record_left"][0],
                        "candidates": v["record_right"],
                        "labels": v["label"],
                    }
                )

            preds_lst = thread_map(
                select,
                instances,
                max_workers=16,
            )
            preds = [pred for preds in preds_lst for pred in preds]
            labels = [label for it in instances for label in it["labels"]]

            results[dataset] = classification_report(
                labels[: len(preds)], preds, output_dict=True
            )["True"]
            results[dataset].pop("support")
            for k, v in results[dataset].items():
                results[dataset][k] = v * 100
            results[dataset]["cost"] = ACCUMULATED_COST
            ACCUMULATED_COST = 0

        results["mean"] = {
            "precision": sum(v["precision"] for v in results.values()) / len(results),
            "recall": sum(v["recall"] for v in results.values()) / len(results),
            "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
            "cost": sum(v["cost"] for v in results.values()) / len(results),
        }
        df = pd.DataFrame.from_dict(results, orient="index")
        print(f"[bold red]{pos}[/bold red]")
        print(df)
        print(df.to_csv(float_format="%.2f", index=False))