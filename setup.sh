#!/bin/bash
# setup.sh - Setup script for Bitcoin Price Prediction project

echo "Setting up Bitcoin Price Prediction environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
echo "Then run: python main.py full"