#!/bin/bash

# Setup and Run Script for Smart Mobility AI

echo "🚀 Starting Setup for Smart Mobility AI..."

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found. Please install Python 3."
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. Install Dependencies
echo "⬇️  Installing dependencies..."
# Try installing strict requirements first
if pip install -r requirements_optimized.txt; then
    echo "✅ Dependencies installed successfully."
else
    echo "⚠️  Standard installation failed (likely dlib). Retrying without dlib..."
    # Create temp requirements without dlib
    grep -v "dlib" requirements_optimized.txt > requirements_temp.txt
    pip install -r requirements_temp.txt
    rm requirements_temp.txt
    echo "✅ Core dependencies installed (Fatigue detection will use fallback mode)."
fi

# 5. Create Models Directory (if missing)
if [ ! -d "models" ]; then
    echo "📂 Creating models directory..."
    mkdir -p models
    echo "⚠️  NOTE: You need to download model files into the 'models' directory for full functionality."
    echo "    For now, the system will run in 'Warning Mode' and skip missing models."
fi

# 6. Run the Server
echo "✅ Setup complete!"
echo "🚀 Starting API Server..."
echo "Running on: http://localhost:8000"
echo "API Docs:   http://localhost:8000/docs"
echo "---------------------------------------------------"

# Run with uvicorn (ensure path is correct)
uvicorn ml_backend.api.main:app --host 0.0.0.0 --port 8000 --reload
