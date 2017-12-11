#!/bin/bash

echo "Linting repository..."
source activate decode

flake8 setup.py
flake8 tree_decode --filename=*.py
