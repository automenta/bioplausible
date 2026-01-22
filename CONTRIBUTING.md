# Contributing to Bioplausible

Thank you for your interest in contributing to Bioplausible!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/bioplausible.git`
3. Install dependencies: `pip install -e .[dev]`

## Development Workflow

1. Create a new branch for your feature or fix.
2. Write code and tests.
3. Run tests: `pytest`
4. Format code: `black bioplausible bioplausible_ui`
5. Submit a Pull Request.

## Testing

We use `pytest`. Please ensure all tests pass before submitting a PR.
- Run all tests: `pytest`
- Run fast validation: `eqprop-verify --quick`

## Code Style

- We use `black` for formatting.
- We use `isort` for import sorting.
- We use `flake8` for linting.

## Reporting Issues

Please check existing issues before opening a new one. Provide a clear description and reproduction steps.
