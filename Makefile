# Convenience targets.  Every command goes through uv so the lockfile is
# honoured; see docs/verification.md for what each level checks.
UV ?= uv

.PHONY: help verify quick test lint smoke dry-run check-outputs

help:
	@echo "make verify        full verification: data + lint + tests + smoke + dry run"
	@echo "make quick         fast verification: data + lint + tests"
	@echo "make test          pytest only"
	@echo "make lint          ruff only"
	@echo "make smoke         end-to-end smoke test on the real data (~1 min)"
	@echo "make dry-run       print the exact reproduction commands"
	@echo "make check-outputs validate a completed outputs/ tree"

verify:
	$(UV) run python scripts/verify.py

quick:
	$(UV) run python scripts/verify.py --quick

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check

smoke:
	$(UV) run python scripts/smoke_test.py

dry-run:
	$(UV) run python scripts/run_experiments.py --dry-run

check-outputs:
	$(UV) run python scripts/check_outputs.py
