#!/bin/bash

# Quick test script with fewer epochs for rapid experimentation
# Use this to quickly test different loss weight configurations

MODEL="Stanford_armadillo"
EPOCHS=100  # Fewer epochs for quick testing
BATCH_SIZE=4096
LR=0.001
HIDDEN_DIM=64
NUM_LAYERS=3
MAX_KNOTS=64
MAX_SEG_INSERTIONS=16
MESH_SAVE_INTERVAL=50

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="quick_tests/${TIMESTAMP}"
mkdir -p "${EXPERIMENT_DIR}"

# Log file
LOG_FILE="${EXPERIMENT_DIR}/tests.log"

echo "Starting quick tests at $(date)" | tee -a "${LOG_FILE}"
echo "-------------------------------------------" | tee -a "${LOG_FILE}"

# Function to run a single test
run_test() {
    local test_name=$1
    local weight_segment=$2
    local weight_normal=$3
    local weight_eikonal=$4
    
    echo "" | tee -a "${LOG_FILE}"
    echo "Test: ${test_name}" | tee -a "${LOG_FILE}"
    echo "  Weights: seg=${weight_segment}, norm=${weight_normal}, eik=${weight_eikonal}" | tee -a "${LOG_FILE}"
    
    local save_dir="${EXPERIMENT_DIR}/${test_name}"
    
    python -m neural_spline.main \
        --model "${MODEL}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --hidden-dim ${HIDDEN_DIM} \
        --num-layers ${NUM_LAYERS} \
        --max-knots ${MAX_KNOTS} \
        --max-seg-insertions ${MAX_SEG_INSERTIONS} \
        --mesh-save-interval ${MESH_SAVE_INTERVAL} \
        --weight-segment ${weight_segment} \
        --weight-normal ${weight_normal} \
        --weight-eikonal ${weight_eikonal} \
        --save-dir "${save_dir}" \
        --skip-connections \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success" | tee -a "${LOG_FILE}"
    else
        echo "  ✗ Failed" | tee -a "${LOG_FILE}"
    fi
    echo "-------------------------------------------" | tee -a "${LOG_FILE}"
}

# Quick tests with different weight configurations
run_test "baseline" 1.0 0.0 0.0
run_test "original" 1.0 1e-3 1e-6
run_test "strong_normal" 1.0 1e-2 1e-6
run_test "strong_both" 1.0 1e-2 1e-4

echo "" | tee -a "${LOG_FILE}"
echo "Quick tests completed at $(date)" | tee -a "${LOG_FILE}"
echo "Results in: ${EXPERIMENT_DIR}" | tee -a "${LOG_FILE}"
