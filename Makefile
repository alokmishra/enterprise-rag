.PHONY: help setup install dev run test test-unit test-integration test-e2e test-coverage lint format type-check clean docker-build docker-run infra-up infra-down migrate eval

# Default target
help:
	@echo "Enterprise RAG - Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Complete development setup"
	@echo "  make install        - Install dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Run the application"
	@echo "  make dev            - Run with auto-reload"
	@echo "  make worker         - Run Celery worker"
	@echo "  make shell          - Open IPython shell"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-e2e       - Run end-to-end tests"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linter"
	@echo "  make format         - Format code"
	@echo "  make type-check     - Run type checker"
	@echo "  make check          - Run all checks"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-up       - Start infrastructure (Docker)"
	@echo "  make infra-down     - Stop infrastructure"
	@echo "  make migrate        - Run database migrations"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run in Docker"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval           - Run evaluation suite"
	@echo "  make eval-report    - Generate evaluation report"

# =============================================================================
# Setup & Installation
# =============================================================================

setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Copying environment template..."
	cp -n .env.example .env || true
	@echo "Setup complete! Run 'make infra-up' to start infrastructure."

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

# =============================================================================
# Development
# =============================================================================

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	celery -A src.workers.celery_app worker --loglevel=info

worker-beat:
	celery -A src.workers.celery_app beat --loglevel=info

shell:
	ipython -i -c "from src.app import create_app; app = create_app()"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --ff

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

type-check:
	mypy src/

check: lint format-check type-check
	@echo "All checks passed!"

# =============================================================================
# Infrastructure
# =============================================================================

infra-up:
	docker-compose up -d

infra-down:
	docker-compose down

infra-logs:
	docker-compose logs -f

infra-reset:
	docker-compose down -v
	docker-compose up -d

migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-down:
	alembic downgrade -1

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t enterprise-rag:latest -f deployments/docker/Dockerfile .

docker-run:
	docker run -p 8000:8000 --env-file .env enterprise-rag:latest

docker-push:
	docker push enterprise-rag:latest

# =============================================================================
# CLI Commands
# =============================================================================

ingest:
	@echo "Ingesting documents from $(path)..."
	python -m tools.cli.main ingest $(path)

query:
	@echo "Running query: $(q)"
	python -m tools.cli.main query "$(q)"

# =============================================================================
# Evaluation
# =============================================================================

eval:
	python -m evaluation.scripts.run_evaluation

eval-report:
	python -m evaluation.scripts.run_evaluation --report

eval-benchmark:
	python -m evaluation.benchmarks.run_benchmark

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs serve

docs-build:
	mkdocs build

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

seed-data:
	python scripts/data/seed_sample_data.py

generate-api-docs:
	python tools/generators/generate_docs.py
