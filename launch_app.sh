#!/bin/bash
# SAM-EM Application Launcher — Mac / Linux

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "         SAM-EM  Application Launcher"
echo "============================================"
echo

# ── Locate Miniconda / Anaconda ──────────────────────────────────────────────
CONDA_ROOT=""
for candidate in \
    "$HOME/miniconda3" \
    "$HOME/opt/miniconda3" \
    "/opt/miniconda3" \
    "/opt/homebrew/Caskroom/miniconda/base" \
    "$HOME/anaconda3" \
    "$HOME/opt/anaconda3" \
    "/opt/anaconda3"
do
    if [ -f "$candidate/bin/conda" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    echo "ERROR: Miniconda or Anaconda was not found."
    echo
    echo "Please install Miniconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Found conda at: $CONDA_ROOT"
echo

CONDA_EXE="$CONDA_ROOT/bin/conda"
ENV_DIR="$CONDA_ROOT/envs/SAM-EM-app"
PYTHON_EXE="$ENV_DIR/bin/python"

# ── First-time setup ─────────────────────────────────────────────────────────
if [ ! -d "$ENV_DIR" ]; then
    echo "First-time setup: creating the SAM-EM environment."
    echo "This downloads packages and takes 5-15 minutes."
    echo "Please wait and do NOT close this window."
    echo

    # Mac does not support CUDA — strip CUDA-specific torch lines
    TMP_ENV="$SCRIPT_DIR/_env_mac_tmp.yml"
    sed '/extra-index-url/d; s/torch==.*+cu[0-9]*/torch/; s/torchvision==.*+cu[0-9]*/torchvision/' \
        "$SCRIPT_DIR/environment_app.yml" > "$TMP_ENV"

    "$CONDA_EXE" env create -f "$TMP_ENV"
    EXIT_CODE=$?
    rm -f "$TMP_ENV"

    if [ $EXIT_CODE -ne 0 ]; then
        echo
        echo "ERROR: Failed to create the environment."
        echo "Check that environment_app.yml is in the same folder as this script."
        read -p "Press Enter to exit..."
        exit 1
    fi

    echo
    echo "Setup complete!"
    echo
fi

# ── Check Python exists in environment ───────────────────────────────────────
if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python not found in the SAM-EM environment."
    echo "Delete $ENV_DIR and run this script again."
    read -p "Press Enter to exit..."
    exit 1
fi

# ── Launch the application ───────────────────────────────────────────────────
echo "Starting SAM-EM application..."
echo "You can close this window once the app opens."
echo
cd "$SCRIPT_DIR/application"
"$PYTHON_EXE" app.py
