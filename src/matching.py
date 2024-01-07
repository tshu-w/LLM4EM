from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from openai import OpenAI
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.contrib.concurrent import thread_map

cache = Cache("results/diskcache/matching")


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


if __name__ == "__main__":
    for file in Path("data/llm4em").glob("*.csv"):
        dataset = file.stem
        if dataset == "dblp-scholar":
            continue
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        instances = df.to_dict("records")
        preds = thread_map(
            match,
            instances,
            max_workers=16,
        )
        labels = df["label"]
        print(classification_report(labels[: len(preds)], preds))
        print(confusion_matrix(labels[: len(preds)], preds))

        # fdf = df[df["label"] != preds]
        # fp = fdf.loc[df["label"] == 0]
        # fn = fdf.loc[df["label"] == 1]
        # print("False positives")
        # print(fp)
        # print("False negatives")
        # print(fn)
