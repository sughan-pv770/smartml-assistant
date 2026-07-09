# Contributing to SmartML

Thank you for your interest in contributing to SmartML! We want to make this the most complete and intelligent ML automation library available for Python.

## Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/smartml-contributors/smartml.git
   cd smartml
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[all,dev]"
   ```

## Running Tests

We use `pytest` for testing. Ensure all tests pass before submitting a Pull Request.

```bash
pytest tests/ -v
```

## Code Style

- We use standard Python style guides (PEP 8).
- Ensure your code has Google-style docstrings.
- Provide type hints for all public functions and classes.
- Avoid side effects (e.g., module-level `matplotlib.use("Agg")` or `print()` calls). Use the `logging` module.

## Adding a New Algorithm

To add a new algorithm to the Trainer:
1. Open `smartml/core/trainer.py`.
2. Add the estimator to the dictionary in `_candidate_models`.
3. Add any necessary explanation logic to `_explain_choice`.
4. Run tests to ensure it integrates correctly with the CV leaderboard.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Update the `README.md` or `CHANGELOG.md` if necessary.
4. Open a PR with a clear title and description.
