from __future__ import annotations

from typing import Any, Callable, Dict

from callm.providers.base import Provider
from callm.providers.openai import OpenAIProvider
from callm.providers.cohere import CohereProvider

ProviderFactory = Callable[..., Provider]

_REGISTRY: Dict[str, ProviderFactory] = {
    "openai": lambda **kwargs: OpenAIProvider(**kwargs),
    "cohere": lambda **kwargs: CohereProvider(**kwargs),
}


def get_provider(name: str, **kwargs: Any) -> Provider:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider: {name}")
    return _REGISTRY[key](**kwargs)


def register_provider(name: str, factory: ProviderFactory) -> None:
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Provider already registered: {name}")
    _REGISTRY[key] = factory
