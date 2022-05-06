buildpre:
	poetry run python buildpre.py

.PHONY: build
build:
	poetry run python build.py clean
	poetry run python build.py build_ext --inplace --force

clean:
	rm -rf build
	rm -rf dist

copy:
	poetry run python buildpre.py

.PHONY: dist
dist: copy
	poetry build

docker:
	docker build -t skgrf .

linux: copy docker
	docker run --rm -v $(shell pwd)/dist:/app/dist:rw skgrf build

.PHONY: docs
docs:
	poetry export --without-hashes --dev -f requirements.txt > docs/requirements.txt && \
	cd docs && \
	poetry run sphinx-build -M html . _build -a && \
	cd .. && \
	open docs/_build/html/index.html

fmt:
	poetry run isort .
	poetry run black .

publish: clean sdist
	poetry publish

release: clean sdist
	ghr -u crflynn -r skgrf -c $(shell git rev-parse HEAD) -delete -b "release" -n $(shell poetry version -s) $(shell poetry version -s) dist/*.tar.gz

sdist: copy
	poetry build -f sdist

setup:
	git submodule init
	git submodule update
	asdf install
	poetry install --no-root
	poetry run python buildpre.py
	poetry install

test:
	poetry run pytest --cov=skgrf --cov-report=html tests/
	open htmlcov/index.html

xtest:
	poetry run pytest -v --cov=skgrf -n auto --cov-report=html tests/
	open htmlcov/index.html

dtest:
	docker run -t skgrf run pytest

wheel:
	cibuildwheel --platform linux