#!/usr/bin/env python3
"""
Simple script to plot validation accuracy from JSON data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_val_accuracy(json_path, output_path="val_accuracy.png"):
    """Plot validation accuracy from JSON file."""
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract accuracy data
    dataset_name = list(data.keys())[0]  # Get first dataset
    acc_data = data[dataset_name]
    
    # Convert to lists
    steps = [int(step) for step in acc_data.keys()]
    accuracies = list(acc_data.values())
    
    # Sort by step
    sorted_pairs = sorted(zip(steps, accuracies))
    steps, accuracies = zip(*sorted_pairs)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title(f'Validation Accuracy Evolution - {dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (step, acc) in enumerate(zip(steps, accuracies)):
        plt.annotate(f'{acc:.3f}', (step, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Set y-axis limits with some padding
    y_min, y_max = min(accuracies), max(accuracies)
    y_range = y_max - y_min
    plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation accuracy plot saved to: {output_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Steps: {steps}")
    print(f"Accuracies: {accuracies}")
    print(f"Best accuracy: {max(accuracies):.3f} at step {steps[accuracies.index(max(accuracies))]}")

def main():
    parser = argparse.ArgumentParser(description="Plot validation accuracy from JSON")
    parser.add_argument("json_path", help="Path to JSON file with accuracy data")
    parser.add_argument("--output", "-o", default="val_accuracy.png", 
                       help="Output path for the plot (default: val_accuracy.png)")
    
    args = parser.parse_args()
    
    plot_val_accuracy(args.json_path, args.output)

if __name__ == "__main__":
    main() 