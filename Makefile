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
	@echo "  install          Install the valor_lite library from source."
	@echo "  install-dev      Install valor_lite along with development tools."
	@echo "  pre-commit       Run pre-commit."
	@echo "  tests            Run unit tests."
	@echo "  external-tests   Run external integration tests."
	@echo "  clean            Remove temporary files."
	@echo "  help             Show this help message."