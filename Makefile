.PHONY: install pre-commit tests clean help

install:
	@echo "Installing from source..."
	pip install -e src/[dev]

pre-commit:
	@echo "Running pre-commit..."
	pre-commit install
	pre-commit run --all

test:
	@echo "Running tests..."
	pytest tests/classification
	pytest tests/object_detection
	pytest tests/semantic_segmentation
	pytest tests/text_generation

help:
	@echo "Available targets:"
	@echo "  install          Install from source with developer tools."
	@echo "  pre-commit       Run pre-commit."
	@echo "  tests            Run tests."
	@echo "  help             Show this help message."