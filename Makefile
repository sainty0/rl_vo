PY?=python3
VENV?=.venv

.PHONY: venv install test train_mock

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip

install: venv
	. $(VENV)/bin/activate && pip install -r requirements.txt

test: install
	. $(VENV)/bin/activate && pytest -q

train_mock: install
	DRY_RUN=1 . $(VENV)/bin/activate && $(PY) scripts/train_mock.py
