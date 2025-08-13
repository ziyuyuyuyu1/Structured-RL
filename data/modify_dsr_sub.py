 #!/usr/bin/env python3
"""
Script to modify dsr_sub.parquet for SFT training
Extracts ground_truth from reward_model as response column
"""

import pandas as pd
import numpy as np

def modify_dsr_sub_for_sft():
    # Read the original parquet file
    input_path = "data/train/one_shot_rlvr/dsr_sub.parquet"
    output_path = "data/train/one_shot_rlvr/dsr_sub_sft.parquet"
    
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Extract ground_truth from reward_model column
    print("Extracting ground_truth from reward_model...")
    
    # Handle the reward_model column which contains dictionaries
    def extract_ground_truth(row):
        reward_model = row['reward_model']
        if isinstance(reward_model, dict):
            return reward_model.get('ground_truth', '')
        elif isinstance(reward_model, np.ndarray):
            # Handle numpy array case
            if len(reward_model) > 0 and isinstance(reward_model[0], dict):
                return reward_model[0].get('ground_truth', '')
        return ''
    
    # Create new dataframe with the required structure
    new_df = pd.DataFrame({
        'prompt': df['prompt'],
        'response': df.apply(extract_ground_truth, axis=1),
        'data_source': df['data_source'],
        'ability': df['ability'],
        'extra_info': df['extra_info']
    })
    
    print(f"New shape: {new_df.shape}")
    print(f"New columns: {new_df.columns.tolist()}")
    
    # Show sample data
    print("\nSample data:")
    print("Prompt:", new_df.iloc[0]['prompt'])
    print("Response:", new_df.iloc[0]['response'])
    
    # Save the modified parquet file
    print(f"\nSaving to {output_path}...")
    new_df.to_parquet(output_path, index=False)
    print("Done!")
    
    return output_path

if __name__ == "__main__":
    output_file = modify_dsr_sub_for_sft()
    print(f"\nModified file saved as: {output_file}")
    print("\nNow you can use this file with the standard SFT configuration:")
    print("data.prompt_key=prompt")
    print("data.response_key=response") 