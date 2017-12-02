#!/bin/bash

echo "Creating a Python $PYTHON_VERSION environment"
conda create -n decode python=$PYTHON_VERSION || exit 1
source activate decode

echo "Installing packages..."
conda install scikit-learn flake8 pytest
