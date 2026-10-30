#!/bin/bash

set -e

PROJECT_DIR="$HOME/Projects/ml-course-recommender"
VENV="$PROJECT_DIR/.venv"

echo "=========================================="
echo " ML Course Recommender - Project Setup"
echo "=========================================="

cd "$PROJECT_DIR"

echo ""
echo "[1/4] Checking project structure..."

mkdir -p \
    data/raw \
    data/processed \
    notebooks \
    src/data \
    src/features \
    src/models \
    src/recommenders \
    src/evaluation \
    src/utils \
    tests \
    models \
    reports/figures \
    reports/tables \
    docs

echo "Project directories verified."

echo ""
echo "[2/4] Creating Python virtual environment..."

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[3/4] Upgrading pip..."

"$VENV/bin/python" -m pip install --upgrade pip

echo ""
echo "[4/4] Checking Python environment..."

"$VENV/bin/python" --version
"$VENV/bin/pip" --version

echo ""
echo "=========================================="
echo " Setup completed successfully."
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo ""
echo "source $VENV/bin/activate"
echo ""
