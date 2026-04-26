#!/bin/bash
# Automated Setup Script for Adaptive Knowledge Selector
# Run this script to set up your virtual environment and install dependencies

set -e  # Exit on error

echo "=========================================="
echo "  Adaptive Knowledge Selector - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
echo "   (This may take 5-10 minutes)"
echo ""
pip install -r requirements.txt

echo ""
echo "✅ All dependencies installed!"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
# Add your API keys here

# LLM APIs (at least one required)
GROQ_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=

# Optional
ANTHROPIC_API_KEY=
HUGGINGFACE_TOKEN=
EOF
    echo "✅ .env file created"
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✅ .env file already exists"
fi
echo ""

# Setup Hetionet
echo "🕸️  Setting up Hetionet knowledge graph..."
if [ -f "data/hetionet/hetionet_graph.pkl" ]; then
    echo "✅ Hetionet already set up"
else
    echo "Downloading and processing Hetionet (this takes ~5 minutes)..."
    python scripts/setup_hetionet.py
fi
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python3 << 'PYEOF'
try:
    import torch
    import sentence_transformers
    import networkx
    import pypdf
    print("✅ All core modules imported successfully!")
except ImportError as e:
    print(f"⚠️  Some modules failed to import: {e}")
    print("You may need to install them manually.")
PYEOF
echo ""

# Print summary
echo "=========================================="
echo "  🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your API keys to .env file:"
echo "   nano .env"
echo "   (Add at least GROQ_API_KEY or GEMINI_API_KEY)"
echo ""
echo "3. Test the system:"
echo "   python scripts/test_sources_manually.py"
echo ""
echo "4. When done, deactivate:"
echo "   deactivate"
echo ""
echo "=========================================="
echo ""

# Keep virtual environment activated
exec $SHELL