# Contributing to ChaosNet

Thank you for your interest in contributing to ChaosNet! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](https://github.com/Likara789/chaosnet/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 💡 How Can I Contribute?

### Reporting Bugs
- Check if the bug has already been reported in the [issues](https://github.com/Likara789/chaosnet/issues)
- If not, create a new issue with a clear title and description
- Include steps to reproduce the issue, expected behavior, and actual behavior
- Add code examples or screenshots if applicable

### Feature Requests
- Open an issue with the "enhancement" label
- Describe the feature and why it would be useful
- Include any relevant use cases or examples

### Code Contributions
1. Fork the repository
2. Create a new branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add some amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a pull request

## 🛠 Development Setup

1. Fork and clone the repository
   ```bash
   git clone https://github.com/yourusername/chaosnet.git
   cd chaosnet
   ```

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

## 🔄 Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters
3. Increase the version numbers in any example files and the README.md to the new version that this Pull Request would represent
4. The PR must pass all CI/CD checks before it can be merged
5. You may merge the PR once you have the sign-off of at least one other developer, or if you do not have permission to do that, you may request the reviewer to merge it for you

## 📝 Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Document all public functions and classes with docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 100 characters when possible
- Use `black` for code formatting
- Use `isort` for import sorting

## 🧪 Testing

- Write tests for all new functionality
- Run the test suite before submitting a PR:
  ```bash
  pytest tests/
  ```
- Aim for at least 80% test coverage

## 📚 Documentation

- Update the documentation for any new features or changes
- Keep docstrings up to date
- Add examples for new features

## 📜 License

By contributing, you agree that your contributions will be licensed under its AGPL-3.0 License.
