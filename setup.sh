#!/bin/bash

# Stop script on first error
set -e

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "🔁 Activating virtual environment..."
source venv/bin/activate

echo "⬇️ Installing required Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "✅ Setup complete. To activate the environment later, run:"
echo "   source venv/bin/activate"
