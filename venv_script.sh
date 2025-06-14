#!/bin/bash

# Create a virtual environment
sudo python3 -m venv icp-env
source icp-env/bin/activate

echo "✅ Virtual environment 'icp-env' activated."

# Upgrade pip
pip install --upgrade pip

# Install specific compatible versions
pip install numpy==1.26.4 opencv-python open3d

echo "✅ Installed numpy==1.26.4, opencv-python, and open3d"

# Test that all modules import correctly
python -c "import numpy; import cv2; import open3d; print('✅ All modules imported successfully')"

echo "✅ Your ICP development environment is ready."
echo "To activate it again later: source icp-env/bin/activate"
