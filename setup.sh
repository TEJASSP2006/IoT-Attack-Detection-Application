#!/bin/bash

# IoT Attack Detection - Quick Start Script
# This script helps you set up and run the application

echo "=========================================="
echo "IoT Attack Detection Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python 3 is installed"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Could not create virtual environment"
    echo "Continuing without virtual environment..."
fi

echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed successfully"
echo ""

# Instructions
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download the CIC IoT 2023 dataset:"
echo "   https://www.unb.ca/cic/datasets/iotdataset-2023.html"
echo ""
echo "2. Preprocess the data (optional):"
echo "   python data_preprocessor.py"
echo ""
echo "3. Train the model:"
echo "   python iot_attack_detector.py"
echo ""
echo "4. Launch the web dashboard:"
echo "   python web_dashboard.py"
echo ""
echo "5. Open your browser to:"
echo "   http://localhost:5000"
echo ""
echo "=========================================="
echo ""
echo "For more information, see README.md"
echo ""
