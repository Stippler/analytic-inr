#!/bin/bash

# Script to run multiple training configurations with different loss weights
# This explores different combinations of segment, normal, and eikonal loss weights

MODEL="Stanford_armadillo"
EPOCHS=500
BATCH_SIZE=4096
LR=0.001
HIDDEN_DIM=64
NUM_LAYERS=3
MAX_KNOTS=64
MAX_SEG_INSERTIONS=16
MESH_SAVE_INTERVAL=100

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="experiments/${TIMESTAMP}"
mkdir -p "${EXPERIMENT_DIR}"

# Log file
LOG_FILE="${EXPERIMENT_DIR}/experiments.log"

echo "Starting experiments at $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "-------------------------------------------" | tee -a "${LOG_FILE}"

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local weight_segment=$2
    local weight_normal=$3
    local weight_eikonal=$4
    local hidden_dim=${5:-${HIDDEN_DIM}}  # Use default if not provided
    local num_layers=${6:-${NUM_LAYERS}}  # Use default if not provided
    
    echo "" | tee -a "${LOG_FILE}"
    echo "Running experiment: ${exp_name}" | tee -a "${LOG_FILE}"
    echo "  Weights: segment=${weight_segment}, normal=${weight_normal}, eikonal=${weight_eikonal}" | tee -a "${LOG_FILE}"
    echo "  Architecture: hidden_dim=${hidden_dim}, num_layers=${num_layers}" | tee -a "${LOG_FILE}"
    
    local save_dir="${EXPERIMENT_DIR}/${exp_name}"
    
    python -m neural_spline.main \
        --model "${MODEL}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --hidden-dim ${hidden_dim} \
        --num-layers ${num_layers} \
        --max-knots ${MAX_KNOTS} \
        --max-seg-insertions ${MAX_SEG_INSERTIONS} \
        --mesh-save-interval ${MESH_SAVE_INTERVAL} \
        --weight-segment ${weight_segment} \
        --weight-normal ${weight_normal} \
        --weight-eikonal ${weight_eikonal} \
        --save-dir "${save_dir}" \
        --skip-connections \
        2>&1 | tee -a "${LOG_FILE}"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Completed successfully" | tee -a "${LOG_FILE}"
    else
        echo "  ✗ Failed with exit code ${exit_code}" | tee -a "${LOG_FILE}"
    fi
    echo "-------------------------------------------" | tee -a "${LOG_FILE}"
}

# Baseline: segment loss only
run_experiment "baseline_segment_only" 1.0 0.0 0.0

# Original configuration
run_experiment "original" 1.0 1e-3 1e-6

# Experiment 1: Stronger normal loss
run_experiment "exp1_strong_normal" 1.0 1e-2 1e-6

# Experiment 2: Very strong normal loss
run_experiment "exp2_very_strong_normal" 1.0 1e-1 1e-6

# Experiment 3: Weaker normal loss
run_experiment "exp3_weak_normal" 1.0 1e-4 1e-6

# Experiment 4: Strong eikonal
run_experiment "exp4_strong_eikonal" 1.0 1e-3 1e-4

# Experiment 5: Very strong eikonal
run_experiment "exp5_very_strong_eikonal" 1.0 1e-3 1e-3

# Experiment 6: Balanced losses
run_experiment "exp6_balanced" 1.0 1e-2 1e-4

# Experiment 7: Normal dominant
run_experiment "exp7_normal_dominant" 0.1 1.0 1e-4

# Experiment 8: All equal
run_experiment "exp8_all_equal" 1.0 1.0 1.0

# Experiment 9: Strong normal and eikonal
run_experiment "exp9_strong_normal_eikonal" 1.0 1e-1 1e-3

# Experiment 10: Weak normal, strong eikonal
run_experiment "exp10_weak_normal_strong_eikonal" 1.0 1e-5 1e-3

echo "" | tee -a "${LOG_FILE}"
echo "===========================================" | tee -a "${LOG_FILE}"
echo "ARCHITECTURE EXPERIMENTS" | tee -a "${LOG_FILE}"
echo "===========================================" | tee -a "${LOG_FILE}"
echo "Testing different network architectures (width and depth)" | tee -a "${LOG_FILE}"
echo "Using baseline loss weights: segment=1.0, normal=1e-3, eikonal=1e-6" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Architecture Experiment 1: Smaller network (narrow and shallow)
run_experiment "arch1_small_32x2" 1.0 1e-3 1e-6 32 2

# Architecture Experiment 2: Small but standard depth
run_experiment "arch2_narrow_32x3" 1.0 1e-3 1e-6 32 3

# Architecture Experiment 3: Very narrow but deeper
run_experiment "arch3_very_narrow_deep_32x4" 1.0 1e-3 1e-6 32 4

# Architecture Experiment 4: Wider network (baseline depth)
run_experiment "arch4_wide_128x3" 1.0 1e-3 1e-6 128 3

# Architecture Experiment 5: Very wide network
run_experiment "arch5_very_wide_256x3" 1.0 1e-3 1e-6 256 3

# Architecture Experiment 6: Wide but shallow
run_experiment "arch6_wide_shallow_128x2" 1.0 1e-3 1e-6 128 2

# Architecture Experiment 7: Deeper network (baseline width)
run_experiment "arch7_deeper_64x4" 1.0 1e-3 1e-6 64 4

# Architecture Experiment 8: Much deeper network
run_experiment "arch8_very_deep_64x5" 1.0 1e-3 1e-6 64 5

# Architecture Experiment 9: Very deep but narrow
run_experiment "arch9_narrow_very_deep_32x6" 1.0 1e-3 1e-6 32 6

# Architecture Experiment 10: Wide and deep
run_experiment "arch10_wide_deep_128x4" 1.0 1e-3 1e-6 128 4

# Architecture Experiment 11: Very wide and deep
run_experiment "arch11_very_wide_deep_256x4" 1.0 1e-3 1e-6 256 4

# Architecture Experiment 12: Extreme capacity
run_experiment "arch12_extreme_256x5" 1.0 1e-3 1e-6 256 5

echo "" | tee -a "${LOG_FILE}"
echo "All experiments completed at $(date)" | tee -a "${LOG_FILE}"
echo "Results saved to: ${EXPERIMENT_DIR}" | tee -a "${LOG_FILE}"

# Create a summary file
SUMMARY_FILE="${EXPERIMENT_DIR}/summary.txt"
echo "Experiment Summary" > "${SUMMARY_FILE}"
echo "==================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Configuration:" >> "${SUMMARY_FILE}"
echo "  Model: ${MODEL}" >> "${SUMMARY_FILE}"
echo "  Epochs: ${EPOCHS}" >> "${SUMMARY_FILE}"
echo "  Batch size: ${BATCH_SIZE}" >> "${SUMMARY_FILE}"
echo "  Learning rate: ${LR}" >> "${SUMMARY_FILE}"
echo "  Default Hidden dim: ${HIDDEN_DIM}" >> "${SUMMARY_FILE}"
echo "  Default Num layers: ${NUM_LAYERS}" >> "${SUMMARY_FILE}"
echo "  Max knots: ${MAX_KNOTS}" >> "${SUMMARY_FILE}"
echo "  Max seg insertions: ${MAX_SEG_INSERTIONS}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "LOSS WEIGHT EXPERIMENTS (architecture: 64x3)" >> "${SUMMARY_FILE}"
echo "=============================================" >> "${SUMMARY_FILE}"
echo "  1. baseline_segment_only    : segment=1.0,  normal=0.0,   eikonal=0.0" >> "${SUMMARY_FILE}"
echo "  2. original                 : segment=1.0,  normal=1e-3,  eikonal=1e-6" >> "${SUMMARY_FILE}"
echo "  3. exp1_strong_normal       : segment=1.0,  normal=1e-2,  eikonal=1e-6" >> "${SUMMARY_FILE}"
echo "  4. exp2_very_strong_normal  : segment=1.0,  normal=1e-1,  eikonal=1e-6" >> "${SUMMARY_FILE}"
echo "  5. exp3_weak_normal         : segment=1.0,  normal=1e-4,  eikonal=1e-6" >> "${SUMMARY_FILE}"
echo "  6. exp4_strong_eikonal      : segment=1.0,  normal=1e-3,  eikonal=1e-4" >> "${SUMMARY_FILE}"
echo "  7. exp5_very_strong_eikonal : segment=1.0,  normal=1e-3,  eikonal=1e-3" >> "${SUMMARY_FILE}"
echo "  8. exp6_balanced            : segment=1.0,  normal=1e-2,  eikonal=1e-4" >> "${SUMMARY_FILE}"
echo "  9. exp7_normal_dominant     : segment=0.1,  normal=1.0,   eikonal=1e-4" >> "${SUMMARY_FILE}"
echo " 10. exp8_all_equal           : segment=1.0,  normal=1.0,   eikonal=1.0" >> "${SUMMARY_FILE}"
echo " 11. exp9_strong_normal_eikonal: segment=1.0, normal=1e-1,  eikonal=1e-3" >> "${SUMMARY_FILE}"
echo " 12. exp10_weak_normal_strong_eikonal: segment=1.0, normal=1e-5, eikonal=1e-3" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "ARCHITECTURE EXPERIMENTS (losses: seg=1.0, norm=1e-3, eik=1e-6)" >> "${SUMMARY_FILE}"
echo "================================================================" >> "${SUMMARY_FILE}"
echo " 13. arch1_small_32x2             : 32 hidden, 2 layers (small & shallow)" >> "${SUMMARY_FILE}"
echo " 14. arch2_narrow_32x3            : 32 hidden, 3 layers (narrow)" >> "${SUMMARY_FILE}"
echo " 15. arch3_very_narrow_deep_32x4  : 32 hidden, 4 layers (narrow & deep)" >> "${SUMMARY_FILE}"
echo " 16. arch4_wide_128x3             : 128 hidden, 3 layers (wide)" >> "${SUMMARY_FILE}"
echo " 17. arch5_very_wide_256x3        : 256 hidden, 3 layers (very wide)" >> "${SUMMARY_FILE}"
echo " 18. arch6_wide_shallow_128x2     : 128 hidden, 2 layers (wide & shallow)" >> "${SUMMARY_FILE}"
echo " 19. arch7_deeper_64x4            : 64 hidden, 4 layers (deeper)" >> "${SUMMARY_FILE}"
echo " 20. arch8_very_deep_64x5         : 64 hidden, 5 layers (very deep)" >> "${SUMMARY_FILE}"
echo " 21. arch9_narrow_very_deep_32x6  : 32 hidden, 6 layers (narrow & very deep)" >> "${SUMMARY_FILE}"
echo " 22. arch10_wide_deep_128x4       : 128 hidden, 4 layers (wide & deep)" >> "${SUMMARY_FILE}"
echo " 23. arch11_very_wide_deep_256x4  : 256 hidden, 4 layers (very wide & deep)" >> "${SUMMARY_FILE}"
echo " 24. arch12_extreme_256x5         : 256 hidden, 5 layers (extreme capacity)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Check individual experiment directories for loss plots and meshes." >> "${SUMMARY_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Summary saved to: ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
