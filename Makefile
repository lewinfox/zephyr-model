.PHONY: all
all: format check

.PHONY: check
check:
	@uv run ruff check

.PHONY: format
format:
	@uv run ruff check --select I --fix
	@uv run ruff format
