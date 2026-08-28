# =========================
# run.sh
# =========================
#!/bin/bash

echo "Starting Hotel Service Level ML Pipeline..."

#export MODEL_TYPE=${1:-logistic}

echo "flush bash system..."
rundll32.exe advapi32.dll,ProcessIdleTasks

# =========================
# Check Python environment
# =========================

MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=14

PYTHON_CMD=""

if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
fi

PYTHON_OK=false

if [ -n "$PYTHON_CMD" ]; then

    PYTHON_VERSION=$("$PYTHON_CMD" -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    echo "Detected Python version: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -gt "$MIN_PYTHON_MAJOR" ] || \
       { [ "$PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && \
         [ "$PYTHON_MINOR" -ge "$MIN_PYTHON_MINOR" ]; }; then

        PYTHON_OK=true
        echo "[OK] Python $PYTHON_VERSION satisfies Python >= 3.14"

    else
        echo "[WARNING] Python $PYTHON_VERSION is too old."
    fi
else
    echo "[WARNING] Python is not installed."
fi

# =========================
# Install Python if needed
# =========================

if [ "$PYTHON_OK" = false ]; then

    echo ""
    echo "Python 3.14 or newer is required."
    echo "Installing Python 3.14..."

    if ! command -v winget >/dev/null 2>&1; then
        echo "[ERROR] winget is not available."
        echo "Please install Python 3.14 manually."
        exit 1
    fi

    winget install \
        --id Python.Python.3.14 \
        --exact \
        --source winget \
        --accept-source-agreements \
        --accept-package-agreements

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Python 3.14."
        exit 1
    fi

    echo "[OK] Python 3.14 installation completed."

    # Refresh command lookup
    hash -r 2>/dev/null

    # Locate Python again
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    fi

    if [ -z "$PYTHON_CMD" ]; then
        echo "[ERROR] Python cannot be found after installation."
        echo "Please close and reopen the terminal, then run run.sh again."
        exit 1
    fi
fi

# =========================
# Python information
# =========================

echo ""
echo "========================================="
echo "Python Environment"
echo "========================================="

echo "Python version:"
"$PYTHON_CMD" --version

echo ""
echo "Python executable:"
"$PYTHON_CMD" -c "import sys; print(sys.executable)"

echo ""
echo "Python detailed version:"
"$PYTHON_CMD" -c "import sys; print(sys.version)"

echo ""
echo "Pip version:"
"$PYTHON_CMD" -m pip --version

echo ""

# =========================
# Database
# =========================


DB_PATH="data/hotel.db"
DB_URL="https://techassessment.blob.core.windows.net/aiap25-assessment-data/hotel.db"

if [ ! -f "$DB_PATH" ]; then
    echo "delivery.db not found — downloading from assessment blob storage..."
    mkdir -p data
    curl --fail --location --show-error --output "$DB_PATH" "$DB_URL"
    echo "Download complete: $(du -h "$DB_PATH" | cut -f1)"
else
    echo "hotel.db already present ($(du -h "$DB_PATH" | cut -f1)) — skipping download."
fi


# =========================
# Install dependencies
# =========================


echo "Installing dependencies..."
pip install -r requirements.txt


# =========================
# Run application
# =========================


echo "Running python script - app.py..."
python src/app.py #--model $MODEL_TYPE