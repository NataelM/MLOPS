# Makefile — Pipeline CI/CD Local + Pre-producción
# Uso: make <target>
# Docs: make help

.PHONY: all lint test train validate docker \
        preprod-up preprod-down preprod-logs preprod-ps \
        smoke deploy rollback clean help

COMPOSE_FILE = docker-compose.preprod.yml
VERSION      ?= latest

# ── Pipeline CI/CD local (Tema 7) ─────────────────────────────────────────────
all: lint test train validate docker
	@echo ""
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

lint:
	@echo "=== Linting con flake8 ==="
	flake8 src/ tests/ api/ --config=setup.cfg
	@echo "Linting OK"

test:
	@echo "=== Tests unitarios con pytest ==="
	pytest tests/ --ignore=tests/smoke -v --tb=short --cov=src --cov-report=term-missing

train:
	@echo "=== Generando datos y entrenando modelo ==="
	python src/generate_data.py
	python src/train_pipeline.py

validate:
	@echo "=== Validando métricas (quality gate) ==="
	python src/validate_model.py

docker:
	@echo "=== Build imagen API ==="
	docker build -t picos-intensidad-api:local .

# ── Pre-producción (Tema 8) ───────────────────────────────────────────────────

## Levantar entorno completo pre-producción (3 servicios)
preprod-up:
	@echo "=== Levantando entorno pre-producción ==="
	@echo "  Nota en Codespace: los puertos 8000 y 5000 aparecerán en la pestaña PORTS"
	docker compose -f $(COMPOSE_FILE) up --build -d
	@echo "  Esperando que los servicios inicien (45 s)..."
	sleep 45
	@echo ""
	@echo "=== Entorno listo ==="
	@echo "  API   : http://localhost:8000   (o URL pública en pestaña PORTS)"
	@echo "  Docs  : http://localhost:8000/docs"
	@echo "  MLflow: http://localhost:5000"

## Bajar entorno pre-producción y eliminar volúmenes
preprod-down:
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "=== Entorno detenido y volúmenes eliminados ==="

## Ver logs en tiempo real de todos los servicios
preprod-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

## Ver logs de un servicio específico: make preprod-logs-api
preprod-logs-api:
	docker compose -f $(COMPOSE_FILE) logs -f api

preprod-logs-trainer:
	docker compose -f $(COMPOSE_FILE) logs -f trainer

preprod-logs-mlflow:
	docker compose -f $(COMPOSE_FILE) logs -f mlflow

## Ver estado de los contenedores
preprod-ps:
	docker compose -f $(COMPOSE_FILE) ps

## Smoke tests del entorno (requiere entorno levantado)
smoke:
	@echo "=== Smoke tests del entorno completo ==="
	pytest tests/smoke/ -v --tb=short

## Flujo completo de despliegue manual
deploy:
	bash deploy.sh $(VERSION)

## Rollback a una versión anterior
rollback:
	@echo "=== Rollback a versión $(VERSION) ==="
	docker compose -f $(COMPOSE_FILE) down
	docker tag picos-intensidad-api:$(VERSION) picos-intensidad-api:latest
	docker compose -f $(COMPOSE_FILE) up -d

## Gestión de versiones MLflow
versions:
	python src/manage_versions.py

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ data/ mlruns/ __pycache__ .coverage htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "=== Tema 7 — CI/CD local ==="
	@echo "  make all           — lint + test + train + validate + docker"
	@echo "  make lint          — linting con flake8"
	@echo "  make test          — tests unitarios (excluye smoke)"
	@echo "  make train         — generar datos + entrenar modelo"
	@echo "  make validate      — quality gate de métricas"
	@echo "  make docker        — build imagen API"
	@echo ""
	@echo "=== Tema 8 — Pre-producción ==="
	@echo "  make preprod-up    — levantar stack completo (3 servicios)"
	@echo "  make preprod-down  — bajar stack y eliminar volúmenes"
	@echo "  make preprod-logs  — ver logs en tiempo real"
	@echo "  make preprod-ps    — ver estado de contenedores"
	@echo "  make smoke         — smoke tests (entorno debe estar levantado)"
	@echo "  make deploy VERSION=v1.2.0  — flujo completo de despliegue"
	@echo "  make rollback VERSION=v1.1.0 — revertir a versión anterior"
	@echo "  make versions      — gestionar versiones MLflow"
	@echo "  make clean         — limpiar artefactos"
	@echo ""