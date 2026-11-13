PYTHON ?= python3  # Python command
PYTEST ?= pytest  # Pytest command
EXTRA_ARGS ?=  # extra arguments for pytest

test:
	$(PYTHON) -m $(PYTEST) $(EXTRA_ARGS)

test-gpu:
	$(PYTHON) -m $(PYTEST) -m gpu $(EXTRA_ARGS)

.PHONY: test test-gpu
