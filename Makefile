PYTHON ?= python
VENV_PYTHON ?= .venv/Scripts/python.exe

.PHONY: venv install lock lint fmt test demo gate verify clean

venv:
	$(PYTHON) -m venv .venv

install: venv
	$(VENV_PYTHON) -m pip install -r requirements.lock.txt
	$(VENV_PYTHON) -m pip install -e .

lock:
	$(VENV_PYTHON) -m pip freeze > requirements.lock.txt

lint:
	$(VENV_PYTHON) -m ruff check .

fmt:
	$(VENV_PYTHON) -m ruff format .

test:
	$(VENV_PYTHON) -m pytest -q

demo:
	$(VENV_PYTHON) examples/run_demo.py

gate:
	$(VENV_PYTHON) benchmarks/gate_benchmark.py --out outputs/gate_result.json

verify: lint
	$(VENV_PYTHON) -m ruff format --check .
	$(VENV_PYTHON) -m pytest -q
	$(VENV_PYTHON) examples/run_demo.py --smoke

clean:
	rm -rf .pytest_cache .ruff_cache artifacts dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
