# Contributing to callm

Thank you for your interest in contributing to callm! This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Making Changes](#making-changes)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding a New Provider](#adding-a-new-provider)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) for dependency management

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/milistu/callm.git
cd callm

# Install dependencies (including dev tools)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Install Python versions for testing
uv python install 3.10 3.11 3.12 3.13 3.14
```

## Architecture Overview

```
src/callm/
├── __init__.py          # Public API exports
├── core/
│   ├── engine.py        # Main processing loop, async workers
│   ├── rate_limit.py    # Token bucket rate limiting
│   ├── retry.py         # Exponential backoff with jitter
│   ├── models.py        # Config dataclasses (RateLimitConfig, etc.)
│   └── io.py            # JSONL read/write utilities
├── providers/
│   ├── base.py          # BaseProvider abstract class
│   ├── openai.py        # OpenAI implementation
│   ├── anthropic.py     # Anthropic implementation
│   ├── gemini.py        # Google Gemini implementation
│   ├── deepseek.py      # DeepSeek implementation
│   ├── cohere.py        # Cohere implementation
│   ├── voyageai.py      # Voyage AI implementation
│   └── registry.py      # Provider name → class mapping
├── tokenizers/
│   ├── openai.py        # tiktoken-based counting
│   ├── anthropic.py     # Claude token estimation
│   └── ...              # Other tokenizer implementations
└── utils.py             # Shared utilities
```

### Key Components

**Core Engine (`core/engine.py`)**
- `process_requests()`: Main entry point for processing
- Async worker loop with rate limiting
- Request queuing and retry scheduling

**Rate Limiting (`core/rate_limit.py`)**
- Token bucket algorithm for RPM and TPM
- Second-level refill for precise throttling

**Providers (`providers/`)**
- Abstract `BaseProvider` class defines the interface
- Each provider implements: `estimate_input_tokens()`, `parse_error()`, `extract_usage()`
- Default implementations for common patterns (Bearer auth, JSON POST)

**Tokenizers (`tokenizers/`)**
- Provider-specific token counting
- Some use local tokenizers (tiktoken), others estimate

## Making Changes

### Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests and linting:
   ```bash
   uv run nox
   ```

4. Commit with a descriptive message:
   ```bash
   git add .
   git commit -m "Add: description of your change"
   ```

### Commit Message Guidelines

- `Add:` New features
- `Fix:` Bug fixes
- `Update:` Changes to existing functionality
- `Docs:` Documentation changes
- `Test:` Test additions or fixes
- `Refactor:` Code restructuring without behavior changes

## Code Quality

### Automatic Checks

Pre-commit hooks run automatically on every commit:

| Tool | Purpose |
|------|---------|
| **Black** | Code formatting |
| **Ruff** | Linting and import sorting |
| **Mypy** | Type checking |

### Manual Commands

```bash
# Run all checks
uv run pre-commit run --all-files

# Auto-format code
uv run nox -s format

# Type check
uv run nox -s type_check
```

### Style Guidelines

- Type hints required for all public functions
- Docstrings for classes and public methods
- Max line length: 100 characters
- Import order: stdlib → third-party → local

## Testing

### Running Tests

```bash
# Run on all Python versions
uv run nox

# Run on specific version
uv run nox -s tests-3.12

# Quick test during development
uv run pytest tests/ -v

# With coverage report
uv run nox -s coverage
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_*.py`
- Name functions `test_*`
- Use pytest fixtures for common setup
- Mock external API calls

Example test structure:

```python
# tests/test_rate_limit.py
import pytest
from callm.core.rate_limit import TokenBucket

def test_token_bucket_consumes_tokens():
    bucket = TokenBucket.start(capacity_per_minute=60)
    assert bucket.try_consume(10) is True
    # ... more assertions
```

## Submitting Changes

1. **Ensure all checks pass:**
   ```bash
   uv run nox
   ```

2. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Link to any related issues
   - Screenshots/examples if applicable

## Adding a New Provider

To add support for a new LLM provider:

### 1. Create Provider Class

```python
# src/callm/providers/newprovider.py
from callm.providers.base import BaseProvider
from callm.providers.models import Usage

class NewProvider(BaseProvider):
    name = "newprovider"

    def __init__(self, api_key: str, model: str, request_url: str):
        self.api_key = api_key
        self.model = model
        self.request_url = request_url

    async def estimate_input_tokens(self, request_json, session=None) -> int:
        # Implement token counting logic
        pass

    def parse_error(self, payload) -> str | None:
        # Extract error message from response
        pass

    def extract_usage(self, payload, estimated_input_tokens=None) -> Usage | None:
        # Extract token usage from response
        pass
```

### 2. Register Provider

```python
# src/callm/providers/__init__.py
from callm.providers.newprovider import NewProvider

__all__ = [..., "NewProvider"]
```

### 3. Add Tokenizer (if needed)

```python
# src/callm/tokenizers/newprovider.py
def count_tokens(text: str, model: str) -> int:
    # Implement token counting
    pass
```

### 4. Create Example

```python
# examples/providers/newprovider/basic.py
# Show basic usage of the new provider
```

### 5. Add Tests

```python
# tests/test_newprovider.py
def test_newprovider_token_counting():
    # Test token counting logic
    pass
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
