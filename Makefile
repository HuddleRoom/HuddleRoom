.PHONY: serve migrate-local migration-local test-local shell-local build-frontend build-and-serve

# ── Local Development ────────────────────────────────────────────────────────

serve:
	.venv/bin/huddleroom serve --reload

migrate-local:
	alembic upgrade head

migration-local:
	alembic revision --autogenerate -m "$(msg)"

test-local:
	.venv/bin/python -m pytest -v -n auto

shell-local:
	python

# ── Frontend ──────────────────────────────────────────────────────────────────

build-frontend:
	cd frontend && npm run build
	mkdir -p huddleroom/static/dashboard
	rm -rf huddleroom/static/dashboard/*
	cp -r frontend/dist/* huddleroom/static/dashboard/

build-and-serve: build-frontend
	$(MAKE) serve