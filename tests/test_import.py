"""Basic import and smoke tests."""


def test_can_import_callm() -> None:
    """Test that the main package can be imported."""
    import callm

    assert callm is not None, "callm package cannot be imported"


def test_can_import_core_modules() -> None:
    """Test that core modules can be imported."""
    from callm.core import engine, models, rate_limit, retry

    assert all([engine, models, rate_limit, retry]), "Core modules cannot be imported"


def test_can_import_providers() -> None:
    """Test that provider modules can be imported."""
    from callm.providers import base, openai, cohere, deepseek, voyageai

    assert all(
        [base, openai, cohere, deepseek, voyageai]
    ), "Provider modules cannot be imported"
