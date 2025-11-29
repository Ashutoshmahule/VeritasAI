from typing import Dict

from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, object] = {}

        self.models["basic"] = ChatOpenAI(
            model="gpt-5.1",
            temperature=0.1,
        )

        self.models["reasoning"] = ChatOpenAI(
            model="gpt-5",
            temperature=0.0,
        )

        self.models["vision"] = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
        )

        self.models["embedding"] = OpenAIEmbeddings(model="text-embedding-3-large")

    def get(self, name: str):
        if name not in self.models:
            raise ValueError(
                f"Unknown model '{name}'. Available models: {list(self.models.keys())}"
            )
        return self.models[name]


model_registry = ModelRegistry()


def basic_model():
    return model_registry.get("basic")


def reasoning_model() -> LanguageModelLike:
    return model_registry.get("reasoning")


def vision_model():
    return model_registry.get("vision")


def embedding_model():
    return model_registry.get("embedding")
