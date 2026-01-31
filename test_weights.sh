#!/bin/bash

# Simple test to verify the weight parameters work correctly
# Runs a very short training to check for errors

echo "Testing loss weight parameters..."

python -m neural_spline.main \
    --model Stanford_armadillo \
    --epochs 2 \
    --batch-size 512 \
    --weight-segment 1.0 \
    --weight-normal 1e-3 \
    --weight-eikonal 1e-6 \
    --max-knots 32 \
    --max-seg-insertions 8 \
    --no-extract-mesh \
    --skip-connections

if [ $? -eq 0 ]; then
    echo "✓ Test passed! Loss weight parameters are working correctly."
else
    echo "✗ Test failed! Check the error messages above."
    exit 1
fi
