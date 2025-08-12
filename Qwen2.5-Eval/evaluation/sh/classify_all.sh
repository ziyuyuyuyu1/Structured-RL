#!/bin/bash

# Script to classify logic sentences for multiple global steps
# Processes global steps 0, 20, 40, ..., 200

BASE_DIR="../../verl_few_shot/Qwen2.5-Math-1.5B-dsr_sub/logic_sentences"
API_KEY="sk-GMqaF5rd7ZrLARbLEKJ1H2vlIpdPQxdOhF27yGC1DkvodBbM"

# Create log directory
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# Function to process a single global step
process_step() {
    local step=$1
    local input_file="$BASE_DIR/global_step_$step/test_qwen25-math-cot_-1_seed0_t0.7_s0_e-1_logic_sentences.jsonl"
    local output_file="$BASE_DIR/global_step_$step/llm_classify.json"
    local log_file="$LOG_DIR/classify_step_${step}.log"
    
    echo "Processing global_step_$step..."
    echo "Input: $input_file"
    echo "Output: $output_file"
    echo "Log: $log_file"
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "ERROR: Input file not found: $input_file" | tee -a "$log_file"
        return 1
    fi
    
    # Check if output directory exists, create if not
    local output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"
    
    # Run the classification with error handling
    echo "Starting classification for step $step at $(date)" | tee -a "$log_file"
    
    if python lm_classify.py \
        --input "$input_file" \
        --output "$output_file" \
        --api-key "$API_KEY" \
        --max-tokens 1024 \
        --timeout 60 \
        --fixed-meanings 2>&1 | tee -a "$log_file"; then
        
        echo "SUCCESS: Completed classification for step $step at $(date)" | tee -a "$log_file"
        echo "Output saved to: $output_file"
        return 0
    else
        echo "ERROR: Classification failed for step $step at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Main execution
echo "Starting batch classification for global steps 0, 20, 40, ..., 200"
echo "Log directory: $LOG_DIR"
echo "Base directory: $BASE_DIR"
echo "=========================================="

# Process each step
for step in 0 20 40 60 80 100 120 140 160 180 200; do
    echo ""
    echo "=========================================="
    echo "Processing global_step_$step"
    echo "=========================================="
    
    if process_step "$step"; then
        echo "✅ Step $step completed successfully"
    else
        echo "❌ Step $step failed - check log: $LOG_DIR/classify_step_${step}.log"
    fi
    
    # Add a small delay between steps to avoid overwhelming the API
    sleep 2
done

echo ""
echo "=========================================="
echo "Batch classification completed!"
echo "Check logs in: $LOG_DIR"
echo "=========================================="

# Summary
echo ""
echo "Summary of results:"
for step in 0 20 40 60 80 100 120 140 160 180 200; do
    log_file="$LOG_DIR/classify_step_${step}.log"
    if [ -f "$log_file" ]; then
        if grep -q "SUCCESS" "$log_file"; then
            echo "✅ Step $step: SUCCESS"
        else
            echo "❌ Step $step: FAILED"
        fi
    else
        echo "❓ Step $step: NO LOG"
    fi
done