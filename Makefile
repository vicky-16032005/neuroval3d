.PHONY: install test smoke benchmark notebooks lint clean

install:
	uv sync

test:
	uv run pytest -q

smoke:
	uv run python scripts/run_smoke.py

benchmark:
	uv run python -m neuroval3d.cli benchmark --synthetic --n-samples 80

notebooks:
	uv run python scripts/generate_notebooks.py

lint:
	uv run ruff check src tests

clean:
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__
