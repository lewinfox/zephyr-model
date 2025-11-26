.PHONY: all
all: format check

.PHONY: check
check:
	@uv run ruff check

.PHONY: format
format:
	@uv run ruff check --select I --fix
	@uv run ruff format

.PHONY: update-db
update-db:
	@uv run --env-file .env python data_preparation/update_db.py
