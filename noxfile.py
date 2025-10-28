"""Nox sessions for testing across multiple Python versions."""

import nox

# Set the default venv backend to uv
nox.options.default_venv_backend = "uv"
# Python versions to test
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
DEFAULT_PYTHON_VERSION = "3.13"

# Nox options
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests with pytest across multiple Python versions."""
    # Install the package and dev dependencies
    session.install(".")
    session.install("pytest", "pytest-asyncio", "pytest-cov")

    # Run tests
    session.run("pytest", "tests/", "-v")


@nox.session(python=PYTHON_VERSIONS)
def coverage(session: nox.Session) -> None:
    """Run tests with coverage report."""
    # Install dependencies
    session.install(".")
    session.install("pytest", "pytest-asyncio", "pytest-cov")

    # Run tests with coverage
    session.run(
        "pytest",
        "tests/",
        "--cov=src/callm",
        "--cov-report=html",
        "--cov-report=term",
        "-v",
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def lint(session: nox.Session) -> None:
    """Run linters (Ruff and Black)."""
    # Install linting tools
    session.install("ruff", "black")

    # Run linters
    session.run("ruff", "check", "src/", "examples/")
    session.run("black", "--check", "src/", "examples/")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def type_check(session: nox.Session) -> None:
    """Run type checking with mypy."""
    # Install mypy and dependencies
    session.install("mypy", "types-requests")
    session.install(".")

    # Run mypy
    session.run("mypy", "src/")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def format(session: nox.Session) -> None:
    """Format code with Black and Ruff."""
    # Install formatting tools
    session.install("ruff", "black")

    # Auto-fix with ruff and format with black
    session.run("ruff", "check", "--fix", "src/", "examples/")
    session.run("black", "src/", "examples/")
