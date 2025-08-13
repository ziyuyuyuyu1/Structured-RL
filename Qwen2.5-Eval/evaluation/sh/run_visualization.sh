#!/bin/bash

# Script to run the classification evolution visualization

echo "Starting classification evolution visualization..."
echo "================================================"

# Run the visualization script
python visualize_classification_evolution.py \
    --base-dir "../../verl_few_shot/Qwen2.5-Math-1.5B-dsr_sub/logic_sentences" \
    --output-dir "../../verl_few_shot/Qwen2.5-Math-1.5B-dsr_sub/logic_sentences/classification_visualizations"

echo ""
echo "Visualization completed!"
echo "Check the 'classification_visualizations' directory for results."
echo "================================================" 