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

def find_matching_problems(dsr_sub_df, original_df):
    """Find matching problems between the subset and original dataset"""
    print("Finding matching problems...")
    
    # Extract problems from the subset
    subset_problems = []
    for idx, row in dsr_sub_df.iterrows():
        problem = extract_problem_from_prompt(row['prompt'])
        if problem:
            subset_problems.append({
                'index': idx,
                'problem': problem,
                'answer': row['reward_model']['ground_truth'],
                'original_row': row
            })
    
    print(f"Found {len(subset_problems)} problems in subset")
    
    # Find matches in original dataset
    matches = []
    for subset_item in subset_problems:
        problem_text = subset_item['problem']
        
        # Look for exact or similar matches
        for orig_idx, orig_row in original_df.iterrows():
            orig_problem = orig_row.get('problem', '')
            if problem_text in orig_problem or orig_problem in problem_text:
                matches.append({
                    'subset_index': subset_item['index'],
                    'original_index': orig_idx,
                    'problem': problem_text,
                    'answer': subset_item['answer'],
                    'solution': orig_row.get('solution', ''),
                    'original_data': orig_row.to_dict()
                })
                break
    
    print(f"Found {len(matches)} matches")
    return matches

def create_sft_dataset(matches):
    """Create SFT dataset with problem, solution, and answer"""
    print("Creating SFT dataset...")
    
    sft_data = []
    for match in matches:
        # Create a comprehensive response that includes solution and answer
        solution = match['solution']
        answer = match['answer']
        
        if solution:
            response = f"{solution}\n\nFinal Answer: {answer}"
        else:
            response = f"Final Answer: {answer}"
        
        sft_data.append({
            'prompt': match['problem'],
            'response': response,
            'answer': answer,
            'solution': solution,
            'subset_index': match['subset_index'],
            'original_index': match['original_index']
        })
    
    return pd.DataFrame(sft_data)

def main():
    # Load the current subset
    print("Loading current subset...")
    dsr_sub_df = pd.read_parquet('data/train/one_shot_rlvr/dsr_sub.parquet')
    print(f"Current subset shape: {dsr_sub_df.shape}")
    
    # Download original dataset
    train_df, test_df = download_deepscaler_dataset()
    if train_df is None:
        print("Failed to download dataset")
        return
    
    # Find matches
    matches = find_matching_problems(dsr_sub_df, train_df)
    
    if len(matches) == 0:
        print("No matches found. Let's examine the data structure...")
        print("\nSample from original dataset:")
        print(train_df.head())
        print("\nSample from subset:")
        print(dsr_sub_df.head())
        return
    
    # Create SFT dataset
    sft_df = create_sft_dataset(matches)
    
    # Save the SFT dataset
    output_path = 'data/train/one_shot_rlvr/dsr_sub_sft_complete.parquet'
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
    simple_output_path = 'data/train/one_shot_rlvr/dsr_sub_sft_simple.parquet'
    simple_sft_df.to_parquet(simple_output_path, index=False)
    print(f"\nSimple SFT dataset saved to: {simple_output_path}")

if __name__ == "__main__":
    main() 