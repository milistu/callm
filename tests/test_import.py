"""Basic import and smoke tests."""


def test_can_import_callm() -> None:
    """Test that the main package can be imported."""
    import callm

    assert callm is not None


def test_can_import_core_modules() -> None:
    """Test that core modules can be imported."""
    from callm.core import engine, models, rate_limit, retry

    assert all([engine, models, rate_limit, retry])


def test_can_import_providers() -> None:
    """Test that provider modules can be imported."""
    from callm.providers import base, openai, cohere, deepseek, voyageai

    assert all([base, openai, cohere, deepseek, voyageai])
