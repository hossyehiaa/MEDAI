#!/usr/bin/env bash
# ==============================================================================
# medAI — One-command setup for fresh clones
# ==============================================================================
# Usage:
#   git clone https://github.com/hossyehiaa/MEDAI.git && cd MEDAI
#   # Add your OpenRouter API key to .env:
#   echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" >> .env
#   ./setup.sh
# ==============================================================================
set -euo pipefail

echo ""
echo "========================================================================"
echo "  medAI — Self-Healing Setup"
echo "========================================================================"
echo ""

# Step 1: Install Python dependencies
echo "  [1/3] Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Step 2: Build search index (ingest PDFs → chunks → embeddings → ChromaDB)
echo "  [2/3] Building search index (ingestion)..."
python ingest.py
echo ""

# Step 3: Run the demo to verify everything works
echo "  [3/3] Running final demo..."
python final_demo.py
echo ""

echo "========================================================================"
echo "  Setup complete! Run:  python final_demo.py"
echo "========================================================================"
echo ""
