# Contributing

Thank you for your interest in contributing to nq-quant-strategy.

This project focuses on quantitative research, systematic trading strategies, and research infrastructure for MNQ futures.

## Before contributing

Please read the project documentation and understand the distinction between:

- Exploratory research
- In-sample results
- Out-of-sample results
- Paper trading
- Live trading

Research claims should be supported by reproducible methodology and clearly stated assumptions.

## Code contributions

When submitting code:

- Follow the existing project structure.
- Keep strategy-specific logic inside the appropriate strategy module.
- Avoid coupling research scripts to the production strategy interface.
- Add tests for new functionality.
- Keep existing tests passing.
- Avoid unnecessary dependencies.
- Document non-obvious quantitative assumptions.

## Research contributions

For new research:

- Clearly state the hypothesis.
- Describe the dataset and time period.
- Specify the methodology.
- Distinguish development data from validation data.
- Avoid optimizing parameters on out-of-sample data.
- Report relevant limitations and failure cases.

## Pull Requests

Pull requests should clearly explain:

1. What changed.
2. Why the change was necessary.
3. How it was tested.
4. Whether the change affects research results or strategy behavior.

For quantitative changes, include relevant validation results where appropriate.

## Issues

Use Issues for:

- Bugs
- Failed tests
- Reproducibility problems
- Feature requests
- Documentation problems

For broader quantitative research questions or ideas, use Discussions.
