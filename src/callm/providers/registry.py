from __future__ import annotations

from collections.abc import Callable
from typing import Any

from callm.providers.base import BaseProvider
from callm.providers.cohere import CohereProvider
from callm.providers.deepseek import DeepSeekProvider
from callm.providers.gemini import GeminiProvider
from callm.providers.openai import OpenAIProvider
from callm.providers.voyageai import VoyageAIProvider

ProviderFactory = Callable[..., BaseProvider]

_REGISTRY: dict[str, ProviderFactory] = {
    "openai": lambda **kwargs: OpenAIProvider(**kwargs),
    "cohere": lambda **kwargs: CohereProvider(**kwargs),
    "voyageai": lambda **kwargs: VoyageAIProvider(**kwargs),
    "deepseek": lambda **kwargs: DeepSeekProvider(**kwargs),
    "gemini": lambda **kwargs: GeminiProvider(**kwargs),
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider: {name}")
    return _REGISTRY[key](**kwargs)


def register_provider(name: str, factory: ProviderFactory) -> None:
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Provider already registered: {name}")
    _REGISTRY[key] = factory
