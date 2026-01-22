# Contributing to Bioplausible

Thank you for your interest in contributing to Bioplausible! We welcome contributions from researchers, developers, and enthusiasts.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bioplausible.git
   cd bioplausible
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .[dev]
   ```
   *Note: Ensure you have PyTorch installed compatible with your hardware.*

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Running Tests

We use `pytest` for testing.

- **Run all tests:**
  ```bash
  python -m pytest
  ```
- **Run tests (headless):**
  ```bash
  export QT_QPA_PLATFORM=offscreen
  python -m pytest
  ```
- **Run verification suite:**
  ```bash
  eqprop-verify --quick
  ```

## Code Style

We follow PEP 8 standards and use `black` for formatting and `isort` for import sorting.

- **Format code:**
  ```bash
  black bioplausible bioplausible_ui
  isort bioplausible bioplausible_ui
  ```
- **Check code quality:**
  ```bash
  flake8 bioplausible bioplausible_ui
  ```

## Pull Request Process

1. Fork the repository and create a new branch for your feature or fix.
2. Ensure all tests pass and code is formatted.
3. Submit a Pull Request with a clear description of your changes.
4. Link any relevant issues.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
