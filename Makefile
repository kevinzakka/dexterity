SHELL := /bin/bash

.PHONY: help check format typecheck test all
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  check: Run type checking and code styling *without* modifying files"
	@echo "  format: Run code styling inplace"
	@echo "  typecheck: Run mypy type checking"
	@echo "  all: Run both type checking and code styling"
	@echo "  test: Run all tests"

check:
	isort --check .
	black --check .
	flake8 --show-source .

format:
	isort .
	black .
	flake8 .

typecheck:
	mypy .

test:
	pytest .

all:
	make format typecheck test
