from collections.abc import Generator
from unittest.mock import Mock

import pytest
import tiktoken
from tokenizers import Tokenizer

from callm.tokenizers import (
    num_tokens_from_deepseek_request,
    num_tokens_from_openai_request,
    num_tokens_from_voyageai_request,
)


class TestOpenAITokenizer:
    """Tests for OpenAI token counting logic."""

    @pytest.fixture
    def tokenizer(self) -> Generator[tiktoken.Encoding, None, None]:
        """Provide a tiktoken encoder for tests."""
        return tiktoken.get_encoding("cl100k_base")

    def test_chat_completions_endpoint(self, tokenizer: tiktoken.Encoding) -> None:
        """Test chat/completions endpoint works and adds message overhead."""
        message = "Hello World"
        request = {"messages": [{"role": "user", "content": message}]}
        num_tokens = len(tokenizer.encode(message))

        num_tokens_estimated = num_tokens_from_openai_request(
            request_json=request,
            api_endpoint="chat/completions",
            tokenizer=tokenizer,
        )

        assert num_tokens_estimated > num_tokens, "Should count message overhead"

    def test_completions_with_string_and_list(self, tokenizer: tiktoken.Encoding) -> None:
        """Test completions handles both string and list prompts."""
        message = "Hello World"
        request_string = {"prompt": message}
        request_list = {"prompt": [message, message]}

        num_tokens_string = len(tokenizer.encode(message))

        tokens_estimated_string = num_tokens_from_openai_request(
            request_json=request_string,
            api_endpoint="completions",
            tokenizer=tokenizer,
        )

        tokens_estimated_list = num_tokens_from_openai_request(
            request_json=request_list,
            api_endpoint="completions",
            tokenizer=tokenizer,
        )

        assert tokens_estimated_string == num_tokens_string, "Should count single string correctly"
        assert (
            tokens_estimated_list == num_tokens_string * 2
        ), "Should count list of strings correctly"

    def test_embeddings_with_string_and_list(self, tokenizer: tiktoken.Encoding) -> None:
        """Test embeddings handles both string and list inputs."""
        message = "Hello World"
        request_string = {"input": message}
        request_list = {"input": [message, message]}

        num_tokens_string = len(tokenizer.encode(message))

        tokens_estimated_string = num_tokens_from_openai_request(
            request_json=request_string,
            api_endpoint="embeddings",
            tokenizer=tokenizer,
        )

        tokens_estimated_list = num_tokens_from_openai_request(
            request_json=request_list,
            api_endpoint="embeddings",
            tokenizer=tokenizer,
        )

        assert tokens_estimated_string == num_tokens_string, "Should count single string correctly"
        assert (
            tokens_estimated_list == num_tokens_string * 2
        ), "Should count list of strings correctly"

    def test_responses_with_string_and_dict(self, tokenizer: tiktoken.Encoding) -> None:
        """Test responses handles string and message objects."""
        message = "Hello World"
        request_string = {"input": message}
        request_dict = {"input": [{"role": "user", "content": message}]}

        num_tokens_string = len(tokenizer.encode(message))

        tokens_estimated_string = num_tokens_from_openai_request(
            request_json=request_string,
            api_endpoint="responses",
            tokenizer=tokenizer,
        )

        tokens_estimated_dict = num_tokens_from_openai_request(
            request_json=request_dict,
            api_endpoint="responses",
            tokenizer=tokenizer,
        )

        assert tokens_estimated_string == num_tokens_string, "Should count single string correctly"
        assert tokens_estimated_dict > num_tokens_string, "Should count message object correctly"

    def test_unsupported_endpoint_raises_error(self, tokenizer: tiktoken.Encoding) -> None:
        """Test that unsupported endpoints raise NotImplementedError."""
        request = {"input": "test"}

        with pytest.raises(NotImplementedError) as exc_info:
            num_tokens_from_openai_request(
                request_json=request,
                api_endpoint="unsupported",
                tokenizer=tokenizer,
            )

        assert "not yet implemented" in str(exc_info.value).lower()

    def test_completions_invalid_prompt_type(self, tokenizer: tiktoken.Encoding) -> None:
        """Test completions raises TypeError for invalid prompt type."""
        request = {"prompt": 123}

        with pytest.raises(TypeError):
            num_tokens_from_openai_request(
                request_json=request,
                api_endpoint="completions",
                tokenizer=tokenizer,
            )

    def test_embeddings_invalid_input_type(self, tokenizer: tiktoken.Encoding) -> None:
        """Test embeddings raises TypeError for invalid input type."""
        request = {"input": {"invalid": "dict"}}

        with pytest.raises(TypeError):
            num_tokens_from_openai_request(
                request_json=request,
                api_endpoint="embeddings",
                tokenizer=tokenizer,
            )

    def test_responses_invalid_input_type(self, tokenizer: tiktoken.Encoding) -> None:
        """Test responses raises TypeError for invalid input type."""
        request = {"input": 123}

        with pytest.raises(TypeError):
            num_tokens_from_openai_request(
                request_json=request,
                api_endpoint="responses",
                tokenizer=tokenizer,
            )


class TestDeepSeekTokenizer:
    """Tests for DeepSeek token counting logic."""

    @pytest.fixture
    def tokenizer(self) -> Tokenizer:
        """Mock tokenizer for testing (DeepSeek holds tokenizers in HuggingFace)."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode = lambda text: list(
            range(len(text.split()))
        )  # Mock: 1 token per word
        return mock_tokenizer

    def test_chat_completions_works(self, tokenizer: Tokenizer) -> None:
        """Test chat completions endpoint works."""
        message = "Hello World"
        request = {"messages": [{"role": "user", "content": message}]}
        num_tokens = len(tokenizer.encode(message))

        num_tokens_estimated = num_tokens_from_deepseek_request(
            request_json=request,
            api_endpoint="chat/completions",
            tokenizer=tokenizer,
        )

        assert num_tokens_estimated > num_tokens, "Should count message overhead"

    def test_completions_string_and_list(self, tokenizer: Tokenizer) -> None:
        """Test completions with string and list prompts."""
        message = "Hello World"
        request_string = {"prompt": message}
        request_list = {"prompt": [message, message]}

        num_tokens_string = len(tokenizer.encode(message))

        tokens_estimated_string = num_tokens_from_deepseek_request(
            request_json=request_string,
            api_endpoint="completions",
            tokenizer=tokenizer,
        )

        tokens_estimated_list = num_tokens_from_deepseek_request(
            request_json=request_list,
            api_endpoint="completions",
            tokenizer=tokenizer,
        )

        assert tokens_estimated_string == num_tokens_string, "Should count single string correctly"
        assert (
            tokens_estimated_list == num_tokens_string * 2
        ), "Should count list of strings correctly"

    def test_unsupported_endpoint_raises_error(self, tokenizer: Tokenizer) -> None:
        """Test unsupported endpoints raise NotImplementedError."""
        request = {"input": "test"}

        with pytest.raises(NotImplementedError) as exc_info:
            num_tokens_from_deepseek_request(
                request_json=request,
                api_endpoint="unsupported",
                tokenizer=tokenizer,
            )

        assert "not yet implemented" in str(exc_info.value).lower()

    def test_invalid_prompt_type_raises_error(self, tokenizer: Tokenizer) -> None:
        """Test invalid prompt type raises TypeError."""
        request = {"prompt": 123}

        with pytest.raises(TypeError):
            num_tokens_from_deepseek_request(
                request_json=request,
                api_endpoint="completions",
                tokenizer=tokenizer,
            )


class TestVoyageAITokenizer:
    """Tests for Voyage AI token counting logic."""

    @pytest.fixture
    def tokenizer(self) -> Tokenizer:
        """Mock tokenizer for testing."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode = lambda text, add_special_tokens=False: list(
            range(len(text.split()))
        )
        return mock_tokenizer

    def test_embeddings_with_string_and_list(self, tokenizer: Tokenizer) -> None:
        """Test embeddings handles both string and list inputs."""
        message = "Hello World"
        request_string = {"input": message}
        request_list = {"input": [message, message]}

        num_tokens_string = len(tokenizer.encode(message))

        tokens_estimated_string = num_tokens_from_voyageai_request(
            request_json=request_string,
            api_endpoint="embeddings",
            tokenizer=tokenizer,
        )

        tokens_estimated_list = num_tokens_from_voyageai_request(
            request_json=request_list,
            api_endpoint="embeddings",
            tokenizer=tokenizer,
        )

        assert tokens_estimated_string == num_tokens_string, "Should count single string correctly"
        assert (
            tokens_estimated_list == num_tokens_string * 2
        ), "Should count list of strings correctly"

    def test_empty_input_returns_zero(self, tokenizer: Tokenizer) -> None:
        """Test empty input returns zero tokens."""
        request = {"input": []}

        tokens_estimated = num_tokens_from_voyageai_request(
            request_json=request,
            api_endpoint="embeddings",
            tokenizer=tokenizer,
        )

        assert tokens_estimated == 0, "Should return zero for empty input"

    def test_unsupported_endpoint_raises_error(self, tokenizer: Tokenizer) -> None:
        """Test unsupported endpoints raise NotImplementedError."""
        request = {"input": "test"}

        with pytest.raises(NotImplementedError) as exc_info:
            num_tokens_from_voyageai_request(
                request_json=request,
                api_endpoint="unsupported",
                tokenizer=tokenizer,
            )

        assert "not yet implemented" in str(exc_info.value).lower()

    def test_invalid_input_type_raises_error(self, tokenizer: Tokenizer) -> None:
        """Test invalid input type raises TypeError."""
        request = {"input": 123}

        with pytest.raises(TypeError):
            num_tokens_from_voyageai_request(
                request_json=request,
                api_endpoint="embeddings",
                tokenizer=tokenizer,
            )
