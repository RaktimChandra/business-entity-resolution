.PHONY: install test run diagnose
install:
	pip install -r requirements.txt pytest
test:
	python -m pytest -q src/tests
run:
	python src/main.py run --data-dir dataset --out-dir output --work-dir work
diagnose:
	python src/tools/diagnose_blocking.py --work-dir work --data-dir dataset --sample 20000
