# Contributing to callm

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) for dependency management

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/callm.git
   cd callm
   ```

2. **Install dependencies**
   ```bash
   uv sync --dev
   ```

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Install Python versions for testing**
   ```bash
   uv python install 3.10 3.11 3.12 3.13 3.14
   ```

## Development Workflow

### Making Changes

1. Create a new branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests locally
   ```bash
   uv run nox
   ```

### Code Quality

Pre-commit hooks automatically run on every commit:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Mypy**: Type checking

To manually run these checks:
```bash
uv run pre-commit run --all-files
```

To auto-format code:
```bash
uv run nox -s format
```

### Testing

**Run tests on all Python versions:**
```bash
uv run nox
```

**Run tests on a specific Python version:**
```bash
uv run nox -s tests-3.13
```

**Check test coverage:**
```bash
uv run nox -s coverage
```

**Run type checking:**
```bash
uv run nox -s type_check
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Run `uv run pytest tests/ -v` for quick testing

## Submitting Changes

1. **Ensure all checks pass**
   ```bash
   uv run nox
   ```

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
   Pre-commit hooks will run automatically.

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub

## Project Structure

```
callm/
├── src/callm/          # Main package code
│   ├── core/           # Core functionality
│   ├── providers/      # LLM provider implementations
│   └── tokenizers/     # Tokenizer implementations
├── tests/              # Test files
├── examples/           # Usage examples
└── noxfile.py          # Test automation config
```

## Questions?

Feel free to open an issue for questions or discussions!
