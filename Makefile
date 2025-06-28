# CUDAgent Makefile
# Provides convenient commands for development and packaging

.PHONY: help install install-dev clean test build dist upload docs lint format

# Default target
help:
	@echo "CUDAgent Development Commands"
	@echo "============================="
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  install-gpu  - Install with GPU support"
	@echo "  install-llm  - Install with LLM support"
	@echo "  install-all  - Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run tests"
	@echo "  test-gpu     - Run GPU tests"
	@echo "  test-llm     - Run LLM tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo ""
	@echo "Packaging:"
	@echo "  build        - Build package"
	@echo "  dist         - Create distribution"
	@echo "  upload       - Upload to PyPI (requires credentials)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo ""
	@echo "CLI Testing:"
	@echo "  cli-test     - Test CLI commands"
	@echo "  cli-setup    - Test setup command"
	@echo "  cli-optimize - Test optimize command"
	@echo "  cli-benchmark- Test benchmark command"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-gpu:
	pip install -e .[gpu]

install-llm:
	pip install -e .[llm]

install-all:
	pip install -e .[all]

# Development targets
test:
	python -m pytest tests/ -v

test-gpu:
	python -m pytest tests/ -m gpu -v

test-llm:
	python -m pytest tests/ -m llm -v

lint:
	flake8 cudagent/ tests/
	mypy cudagent/

format:
	black cudagent/ tests/
	isort cudagent/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Packaging targets
build:
	python setup.py sdist bdist_wheel

dist: clean build
	@echo "Distribution created in dist/"

upload: dist
	twine upload dist/*

# Documentation targets
docs:
	@echo "Building documentation..."
	# Add documentation build commands here when docs are added

docs-serve:
	@echo "Serving documentation..."
	# Add documentation serve commands here when docs are added

# CLI testing targets
cli-test:
	@echo "Testing CLI commands..."
	cudagent --help
	cudagent-setup --help
	cudagent-test --help
	cudagent-optimize --help
	cudagent-benchmark --help

cli-setup:
	@echo "Testing setup command..."
	cudagent setup --help

cli-optimize:
	@echo "Testing optimize command..."
	cudagent optimize --help

cli-benchmark:
	@echo "Testing benchmark command..."
	cudagent benchmark --help

# Quick development workflow
dev: install-dev format lint test

# Full build process
full: clean install-all test build cli-test
	@echo "Full build process completed!"

# Release preparation
release: clean install-all test lint build
	@echo "Release preparation completed!"
	@echo "Generated files in dist/:"
	@ls -la dist/ 