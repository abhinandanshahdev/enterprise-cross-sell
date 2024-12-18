#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

echo "Virtual environment 'venv' created and dependencies installed."
echo "To activate the virtual environment, run: source venv/bin/activate" 