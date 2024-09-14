import math
from pathlib import Path

import nltk
import pandas as pd
from diskcache import Cache
from jinja2 import Template
from retriv import SparseRetriever
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.utils import APICostCalculator, openai_chat_complete

nltk.download = lambda *args, **kwargs: None


def get_retriever(
    dataset: str,
    df: pd.DataFrame,
):
    def gen_docs(df: pd.DataFrame):
        for idx, row in df.iterrows():
            doc = {
                "id": idx,
                "text": f"{row.record_left} {row.record_right}",
                "record_left": row.record_left,
                "record_right": row.record_right,
                "label": row.label,
            }
            yield doc

    try:
        retriever = SparseRetriever.load(f"{dataset}-icl")
    except FileNotFoundError:
        retriever = SparseRetriever(index_name=f"{dataset}-icl")
        retriever = retriever.index(
            gen_docs(df),
            show_progress=True,
        )

    return retriever


class MatchingICL:
    template = Template(
        """Do the two entity records refer to the same real-world entity? Answer "Yes" if they do and "No" if they do not.

Record 1: {{ record_left }}
Record 2: {{ record_right }}
"""
    )

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        template: Template = template,
    ):
        self.model = model_name
        self.template = template

        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(f"results/diskcache/matching_{model_name}")
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )

    def score(self, instance, use_prob: bool = False) -> list[float]:
        scores = []
        for candidate, examples in zip(
            instance["candidates"], instance["examples"], strict=True
        ):
            messages = [
                message
                for example in examples
                for message in [
                    {
                        "role": "user",
                        "content": self.template.render(
                            record_left=example["record_left"],
                            record_right=example["record_right"],
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "Yes" if example["label"] else "No",
                    },
                ]
            ]
            messages.append(
                {
                    "role": "user",
                    "content": self.template.render(
                        record_left=instance["anchor"],
                        record_right=candidate,
                    ),
                }
            )
            response = self.chat_complete(
                messages=messages,
                model=self.model,
                seed=42,
                temperature=0.0,
                logprobs=self.model.startswith("gpt"),
                top_logprobs=3 if self.model.startswith("gpt") else None,
                max_tokens=3,
            )
            if use_prob:
                assert self.model.startswith("gpt")
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

    def pointwise_rank(self, instance) -> list[int]:
        scores = self.score(instance, use_prob=True)
        indexes = list(range(len(instance["candidates"])))
        indexes = [
            x for _, x in sorted(zip(scores, indexes, strict=True), reverse=True)
        ]
        return indexes

    def __call__(
        self,
        instance,
        single_match: bool = False,
    ) -> list[bool]:
        scores = self.score(instance)
        if single_match:
            max_score = max(scores)
            preds = [sc >= max_score and sc > 0 for sc in scores]
        else:
            preds = [sc > 0 for sc in scores]

        return preds

    @property
    def cost(self):
        return self.api_cost_decorator.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    matcher = MatchingICL()
    few_shot = 6
    for file in dataset_files:
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        df_icl = pd.read_csv(Path("data/llm4em/remains") / file.name)
        df_icl_pos = df_icl[df_icl["label"] == True].sample(n=50, random_state=42)  # noqa: E712
        df_icl_neg = df_icl[df_icl["label"] == False].sample(n=50, random_state=42)  # noqa: E712
        df_icl = pd.concat([df_icl_pos, df_icl_neg])
        retriever = get_retriever(dataset, df_icl)

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
                "examples": [
                    retriever.search(
                        f'{v["record_left"][0]} {candidate}',
                        cutoff=few_shot,
                    )
                    for candidate in v["record_right"]
                ],
            }
            for _, v in groupby
        ]

        preds_lst = thread_map(
            matcher,
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {matcher.cost:.2f}")

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
    print(f"{matcher.cost:.2f}")
