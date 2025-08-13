#!/usr/bin/env python3
"""
Script to download DeepScaleR dataset and create SFT dataset
"""

import pandas as pd
import numpy as np
import json
from datasets import load_dataset
import os

def download_deepscaler_dataset():
    """Download the original DeepScaleR dataset"""
    print("Downloading DeepScaleR-Preview-Dataset...")
    
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Convert to pandas for easier processing
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas() if 'test' in dataset else None
        
        print(f"Train set shape: {train_df.shape}")
        if test_df is not None:
            print(f"Test set shape: {test_df.shape}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None

def extract_problem_from_prompt(prompt_array):
    """Extract the problem text from the prompt array"""
    if isinstance(prompt_array, np.ndarray) and len(prompt_array) > 0:
        prompt_dict = prompt_array[0]
        if isinstance(prompt_dict, dict) and 'content' in prompt_dict:
            return prompt_dict['content']
    return None

def process_full_dataset(original_df):
    """Process the full original dataset"""
    print("Processing full original dataset...")
    
    processed_data = []
    for idx, row in original_df.iterrows():
        problem = row.get('problem', '')
        solution = row.get('solution', '')
        answer = row.get('answer', '')
        
        # If no explicit answer field, try to extract from solution
        if not answer and solution:
            # Try to find the final answer in the solution
            lines = solution.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and ('answer' in line.lower() or '=' in line or line.endswith('.') or line.isdigit()):
                    answer = line
                    break
        
        processed_data.append({
            'original_index': idx,
            'problem': problem,
            'solution': solution,
            'answer': answer,
            'original_data': row.to_dict()
        })
    
    print(f"Processed {len(processed_data)} problems from original dataset")
    return processed_data

def create_sft_dataset(processed_data):
    """Create SFT dataset with problem, solution, and answer"""
    print("Creating SFT dataset...")
    
    sft_data = []
    for item in processed_data:
        # Create a comprehensive response that includes solution and answer
        solution = item['solution']
        answer = item['answer']
        
        if solution:
            response = f"{solution}\n\nFinal Answer: {answer}"
        else:
            response = f"Final Answer: {answer}"
        
        sft_data.append({
            'prompt': item['problem'],
            'response': response,
            'answer': answer,
            'solution': solution,
            'original_index': item['original_index']
        })
    
    return pd.DataFrame(sft_data)

def main():
    # Download original dataset
    train_df, test_df = download_deepscaler_dataset()
    if train_df is None:
        print("Failed to download dataset")
        return
    
    # Process full dataset
    processed_data = process_full_dataset(train_df)
    
    if len(processed_data) == 0:
        print("No data processed. Let's examine the data structure...")
        print("\nSample from original dataset:")
        print(train_df.head())
        return
    
    # Create SFT dataset
    sft_df = create_sft_dataset(processed_data)
    
    # Save the SFT dataset
    output_path = 'data/train/one_shot_rlvr/dsr_full_sft_complete.parquet'
    sft_df.to_parquet(output_path, index=False)
    
    print(f"\nSFT dataset created and saved to: {output_path}")
    print(f"Shape: {sft_df.shape}")
    print(f"Columns: {sft_df.columns.tolist()}")
    
    # Show sample
    print("\nSample SFT data:")
    print("Prompt:", sft_df.iloc[0]['prompt'])
    print("Response:", sft_df.iloc[0]['response'])
    
    # Also save a version with just prompt and response for standard SFT
    simple_sft_df = sft_df[['prompt', 'response']].copy()
    simple_output_path = 'data/train/one_shot_rlvr/dsr_full_sft_simple.parquet'
    simple_sft_df.to_parquet(simple_output_path, index=False)
    print(f"\nSimple SFT dataset saved to: {simple_output_path}")

if __name__ == "__main__":
    main() 