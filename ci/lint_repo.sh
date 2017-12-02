#!/bin/bash

echo "Linting repository..."
source activate decode

flake8
