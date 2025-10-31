import pytest

from callm.utils import api_endpoint_from_url, task_id_generator, validate_jsonl_file


class TestAPIEndpointExtraction:
    """Tests for extracting API endpoints from URLs."""

    @pytest.mark.parametrize(
        argnames="url,endpoint",
        argvalues=[
            ("https://api.openai.com/v1/chat/completions", "chat/completions"),
            ("https://api.openai.com/v1/embeddings", "embeddings"),
            (
                "https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
                "chat/completions",
            ),
            ("https://api.deepseek.com/chat/completions", "chat/completions"),
            ("https://api.cohere.com/v2/embed", "embed"),
            ("https://api.voyageai.com/v1/embeddings", "embeddings"),
        ],
    )
    def test_standard_versioned_urls(self, url: str, endpoint: str) -> None:
        """Test extraction from standard versioned API URLs."""

        assert api_endpoint_from_url(url) == endpoint

    def test_invalid_url_raises_error(self) -> None:
        """Test that invalid URLs raise ValueError."""
        with pytest.raises(ValueError):
            api_endpoint_from_url("https://invalid.url")


class TestValidation:
    """Tests for input validation functions."""

    @pytest.mark.parametrize(
        argnames="filepath",
        argvalues=[
            "data/requests.jsonl",
            "output.jsonl",
            "/path/to/file.jsonl",
        ],
    )
    def test_validate_jsonl_file_accepts_valid_extension(self, filepath: str) -> None:
        """Test that .jsonl files pass validation."""
        validate_jsonl_file(filepath)

    @pytest.mark.parametrize(
        argnames="filepath",
        argvalues=[
            "data.json",
            "data.txt",
            "data.csv",
        ],
    )
    def test_validate_jsonl_file_rejects_invalid_extension(self, filepath: str) -> None:
        """Test that non-.jsonl files are rejected."""
        with pytest.raises(ValueError):
            validate_jsonl_file(filepath)


class TestTaskIDGenerator:
    """Tests for task ID generator."""

    def test_generator_produces_sequential_ids(self) -> None:
        """Test that generator produces sequential integers starting from 0."""
        gen = task_id_generator()

        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2
        assert next(gen) == 3
        assert next(gen) == 4

    def test_generator_continues_indefinitely(self) -> None:
        """Test that generator can produce many IDs."""
        gen = task_id_generator()

        # Generate 100 IDs
        ids = [next(gen) for _ in range(100)]

        assert len(ids) == 100
        assert ids == list(range(100))  # Should be [0, 1, 2, ..., 99]
