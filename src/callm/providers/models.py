from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    """
    Token usage metrics from an API response.

    Different providers may report usage differently:
    - OpenAI: prompt_tokens, completion_tokens, total_tokens
    - Anthropic: input_tokens, output_tokens
    - This model normalizes to input/output/total

    Attributes:
        input_tokens (int): Number of tokens in the input/prompt
        output_tokens (int): Number of tokens in the output/completion
        total_tokens (int): Total tokens used (input + output)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
