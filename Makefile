.PHONY: setup fmt lint type test precommit smoke release unlock maintenance-off maintenance-on ci-validate

setup:
	pip install -U pip
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

fmt:
	black ironforge tests

lint:
	ruff check ironforge tests

type:
	mypy ironforge

test:
	pytest -q

ci-validate:
	@echo "Running full CI validation suite..."
	python3 scripts/validate-ci.py

precommit:
	pre-commit run --all-files

smoke:
	@echo "[smoke] Verify package presence"
	test -d ironforge || (echo "ironforge/ package missing" && exit 1)
	@echo "[smoke] Import public modules"
	python3 -c 'import importlib,sys; mods=["ironforge","ironforge.validation","ironforge.reporting","ironforge.confluence","ironforge.sdk"]; [importlib.import_module(m) for m in mods]; print("import smoke OK")'

# Usage: make release VERSION=v0.6.0 MSG="Waves 4–6: Validation Rails, Reporting, Confluence"
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "ERROR: VERSION is required (e.g., make release VERSION=v0.6.0)"; \
		exit 1; \
	fi
	git tag -a $(VERSION) -m "$(if $(MSG),$(MSG),Release $(VERSION))"
	git push origin $(VERSION)
	-which gh >/dev/null 2>&1 && gh release create $(VERSION) --generate-notes || echo "Install GitHub CLI (gh) to auto-create releases"

.PHONY: discover score validate report status
discover:
	python3 -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
score:
	python3 -m ironforge.sdk.cli score-session --config configs/dev.yml
validate:
	python3 -m ironforge.sdk.cli validate-run --config configs/dev.yml
report:
	python3 -m ironforge.sdk.cli report-minimal --config configs/dev.yml
status:
	python3 -m ironforge.sdk.cli status --runs runs

# Write docs/context.json from repo state (non-destructive)
.PHONY: context
context:
	python3 scripts/gen_context.py

unlock:
	./scripts/git-unlock.sh

maintenance-off:
	./scripts/git-maintenance-off.sh

maintenance-on:
	./scripts/git-maintenance-on.sh
