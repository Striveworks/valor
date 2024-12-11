.PHONY: install install-dev pre-commit tests external-tests clean help

install:
	@echo "Installing from source..."
	pip install -e src/[mistral, openai]

install-dev:
	pip install -e src/[all]
	pre-commit install

pre-commit:
	@echo "Running pre-commit..."
	pre-commit run --all

tests:
	@echo "Running unit tests..."
	poetry run pytest ./lite/tests/text_generation -v

external-tests:
	@echo "Running external integration tests..."
	poetry run pytest ./lite/tests/text_generation -v
	poetry run pytest ./integration_tests/external -v

clean:
	@echo "Cleaning up temporary files..."
	rm -rf .pytest_cache __pycache__ valor_lite.egg-info

help:
	@echo "Available targets:"
	@echo "  tests            Run unit tests using pytest."
	@echo "  external-tests   Run external integration tests with verbose output."
	@echo "  clean            Remove temporary files like .pytest_cache and __pycache__."
	@echo "  help             Show this help message."