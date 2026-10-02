# Makefile — Pipeline CI/CD Local
# Simula exactamente los mismos pasos que GitHub Actions antes de hacer push
# Uso: make <target>
# Uso completo: make all

.PHONY: all lint test train validate docker clean help

# ── Target por defecto: ejecutar todo en secuencia ────────────────────────────
all: lint test train validate docker
	@echo ""
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

# ── Linting con flake8 ────────────────────────────────────────────────────────
lint:
	@echo "=== Linting con flake8 ==="
	flake8 src/ tests/ --config=setup.cfg
	@echo "Linting OK"

# ── Tests unitarios con pytest ────────────────────────────────────────────────
test:
	@echo "=== Tests con pytest ==="
	@python -c "import pytest_cov" 2>/dev/null || \
		(echo "ERROR: falta pytest-cov. Ejecuta: pip install -r requirements.txt" && exit 1)
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Generar datos y entrenar modelo ───────────────────────────────────────────
train:
	@echo "=== Generando datos ==="
	python src/generate_data.py
	@echo "=== Entrenando modelo ==="
	python src/train_pipeline.py

# ── Quality gate de métricas ──────────────────────────────────────────────────
validate:
	@echo "=== Validando métricas (quality gate) ==="
	python src/validate_model.py

# ── Build de imagen Docker ────────────────────────────────────────────────────
docker:
	@echo "=== Construyendo imagen Docker ==="
	docker build -t picos-intensidad-api:local .
	@echo "=== Smoke test ==="
	docker run --rm picos-intensidad-api:local python -c \
		"import pickle; m=pickle.load(open('artifacts/modelo.pkl','rb')); print('Modelo OK:', type(m).__name__)"

# ── Limpiar artefactos generados ──────────────────────────────────────────────
clean:
	rm -rf artifacts/ data/ mlruns/ __pycache__ .coverage htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Targets disponibles:"
	@echo "  make all       — Ejecutar todo: lint + test + train + validate + docker"
	@echo "  make lint      — Solo linting con flake8"
	@echo "  make test      — Solo tests con pytest"
	@echo "  make train     — Generar datos + entrenar modelo"
	@echo "  make validate  — Quality gate de métricas"
	@echo "  make docker    — Build y smoke test de la imagen Docker"
	@echo "  make clean     — Limpiar artefactos y cachés"
	@echo ""