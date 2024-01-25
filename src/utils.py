from functools import wraps


class APICostCalculator:
    _model_cost_per_1k_tokens = {
        # https://platform.openai.com/docs/deprecations/
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.0020},
        "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.0020},
        "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.0020},
        "gpt-3.5-turbo-1106": {"prompt": 0.0010, "completion": 0.0020},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        # https://openrouter.ai/docs#models
        "mistralai/mistral-tiny": {"prompt": 0.0001555, "completion": 0.0004666},
        "mistralai/mistral-small": {"prompt": 0.0006666, "completion": 0.002},
        "mistralai/mistral-medium": {"prompt": 0.002778, "completion": 0.008333},
        # https://docs.endpoints.anyscale.com/pricing
        "meta-llama/Llama-2-7b-chat-hf": {"prompt": 0.00015, "completion": 0.00015},
        "meta-llama/Llama-2-13b-chat-hf": {"prompt": 0.00025, "completion": 0.00025},
        "meta-llama/Llama-2-70b-chat-hf": {"prompt": 0.001, "completion": 0.001},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {
            "prompt": 0.0005,
            "completion": 0.0005,
        },
        # https://replicate.com/pricing
        "meta/llama-2-7b": {"prompt": 0.00005, "completion": 0.00025},
        "meta/llama-2-13b": {"prompt": 0.00010, "completion": 0.00050},
        "meta/llama-2-70b": {"prompt": 0.00065, "completion": 0.00275},
        "mistralai/mistral-7b-v0.1": {"prompt": 0.00005, "completion": 0.00025},
        "mistralai/mistral-7b-instruct-v0.2": {
            "prompt": 0.00005,
            "completion": 0.00025,
        },
        "mistralai/mixtral-8x7b-instruct-v0.1": {
            "prompt": 0.00030,
            "completion": 0.00100,
        },
    }

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if model_name not in self._model_cost_per_1k_tokens:
            raise ValueError(f"Unknown model name: {model_name}")
        self._model_name = model_name
        self._cost = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            cost = (
                self._model_cost_per_1k_tokens[self._model_name]["prompt"]
                * response.usage.prompt_tokens
                + self._model_cost_per_1k_tokens[self._model_name]["completion"]
                * response.usage.completion_tokens
            ) / 1000.0
            self._cost += cost
            return response

        return wrapper

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value: int):
        self._cost = value
