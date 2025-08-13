#!/usr/bin/env python3
"""
Visualize the evolution of meaning distributions over training steps.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClassificationEvolutionVisualizer:
    """Visualize the evolution of meaning distributions over training steps."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        # Start with all possible steps, but we'll filter to only those with data
        self.all_steps = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        self.steps = []  # Will be populated with steps that have data
        self.meanings = None
        self.data = {}
        
    def load_data(self):
        """Load all distribution data from the training steps."""
        print("Loading distribution data...")
        
        # First, find which steps have data
        available_steps = []
        for step in self.all_steps:
            step_dir = self.base_dir / f"global_step_{step}"
            dist_file = step_dir / "llm_classify_distributions.json"
            
            if dist_file.exists():
                available_steps.append(step)
                with open(dist_file, 'r') as f:
                    self.data[step] = json.load(f)
                print(f"Found data for step {step}: {self.data[step]['total_sentences_processed']} sentences")
            else:
                print(f"No data found for step {step} - skipping")
        
        self.steps = sorted(available_steps)
        print(f"\nAvailable steps with data: {self.steps}")
        
        # Get meanings from the first available step
        if self.data:
            first_step = min(self.data.keys())
            self.meanings = self.data[first_step]['meanings']
            print(f"Found {len(self.meanings)} meaning categories")
        else:
            print("No data found for any step!")
            return
    
    def create_overall_distribution_evolution(self, output_dir: str = "visualizations"):
        """Create visualization of overall meaning distribution evolution."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Prepare data
        meaning_names = [f"Meaning {i}" for i in range(len(self.meanings))]
        step_data = []
        
        for step in self.steps:
            if step in self.data:
                overall_freq = self.data[step]['overall_frequencies']
                total = sum(overall_freq.values())
                # Convert string keys to int for proper indexing
                percentages = [overall_freq.get(str(i), 0) / total * 100 for i in range(len(self.meanings))]
                step_data.append(percentages)
            else:
                step_data.append([0] * len(self.meanings))
        
        step_data = np.array(step_data)
        
        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create stacked area chart
        ax.stackplot(self.steps, step_data.T, labels=meaning_names, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Percentage of Sentences (%)', fontsize=12)
        ax.set_title('Evolution of Meaning Distribution Over Training Steps', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'meaning_distribution_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved overall distribution evolution to {output_path / 'meaning_distribution_evolution.png'}")
    
    def create_prompt_distribution_heatmaps(self, output_dir: str = "visualizations"):
        """Create heatmaps showing distribution evolution for each prompt."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all prompt IDs
        all_prompts = set()
        for step_data in self.data.values():
            all_prompts.update(step_data['prompt_distributions'].keys())
        all_prompts = sorted(all_prompts, key=int)
        
        # Create subplot for each prompt
        n_prompts = len(all_prompts)
        n_cols = 2
        n_rows = (n_prompts + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, prompt_id in enumerate(all_prompts):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Prepare data for this prompt
            prompt_data = []
            for step in self.steps:
                if step in self.data and prompt_id in self.data[step]['prompt_distributions']:
                    dist = self.data[step]['prompt_distributions'][prompt_id]
                    total = sum(dist.values())
                    # Convert string keys to int for proper indexing
                    percentages = [dist.get(str(i), 0) / total * 100 for i in range(len(self.meanings))]
                    prompt_data.append(percentages)
                else:
                    prompt_data.append([0] * len(self.meanings))
            
            prompt_data = np.array(prompt_data)
            
            # Create heatmap
            im = ax.imshow(prompt_data.T, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(self.steps)))
            ax.set_xticklabels(self.steps, rotation=45)
            ax.set_yticks(range(len(self.meanings)))
            ax.set_yticklabels([f"M{i}" for i in range(len(self.meanings))])
            ax.set_title(f'Prompt {prompt_id} Distribution Evolution', fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Meaning Category')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Percentage (%)')
        
        # Hide empty subplots
        for idx in range(n_prompts, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'prompt_distribution_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved prompt distribution heatmaps to {output_path / 'prompt_distribution_heatmaps.png'}")
    
    def create_prompt_comparison_analysis(self, output_dir: str = "visualizations"):
        """Create detailed prompt comparison analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all prompt IDs
        all_prompts = set()
        for step_data in self.data.values():
            all_prompts.update(step_data['prompt_distributions'].keys())
        all_prompts = sorted(all_prompts, key=int)
        
        # 1. Prompt-wise evolution comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Evolution of most common meaning for each prompt
        ax1 = axes[0, 0]
        for prompt_id in all_prompts:
            prompt_evolution = []
            for step in self.steps:
                if step in self.data and prompt_id in self.data[step]['prompt_distributions']:
                    dist = self.data[step]['prompt_distributions'][prompt_id]
                    total = sum(dist.values())
                    if total > 0:
                        # Find most common meaning
                        most_common = max(dist.items(), key=lambda x: x[1])
                        percentage = most_common[1] / total * 100
                        prompt_evolution.append(percentage)
                    else:
                        prompt_evolution.append(0)
                else:
                    prompt_evolution.append(0)
            
            ax1.plot(self.steps, prompt_evolution, marker='o', linewidth=2, markersize=6, 
                    label=f'Prompt {prompt_id}')
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Most Common Meaning (%)')
        ax1.set_title('Evolution of Most Common Meaning per Prompt')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prompt diversity evolution (number of meanings with >10% usage)
        ax2 = axes[0, 1]
        for prompt_id in all_prompts:
            diversity_evolution = []
            for step in self.steps:
                if step in self.data and prompt_id in self.data[step]['prompt_distributions']:
                    dist = self.data[step]['prompt_distributions'][prompt_id]
                    total = sum(dist.values())
                    if total > 0:
                        percentages = [dist.get(str(i), 0) / total * 100 for i in range(len(self.meanings))]
                        diverse_count = sum(1 for p in percentages if p > 10)
                        diversity_evolution.append(diverse_count)
                    else:
                        diversity_evolution.append(0)
                else:
                    diversity_evolution.append(0)
            
            ax2.plot(self.steps, diversity_evolution, marker='o', linewidth=2, markersize=6, 
                    label=f'Prompt {prompt_id}')
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Number of Meanings >10%')
        ax2.set_title('Prompt Diversity Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prompt entropy evolution
        ax3 = axes[1, 0]
        for prompt_id in all_prompts:
            entropy_evolution = []
            for step in self.steps:
                if step in self.data and prompt_id in self.data[step]['prompt_distributions']:
                    dist = self.data[step]['prompt_distributions'][prompt_id]
                    total = sum(dist.values())
                    if total > 0:
                        probs = [dist.get(str(i), 0) / total for i in range(len(self.meanings))]
                        entropy_evolution.append(entropy(probs))
                    else:
                        entropy_evolution.append(0)
                else:
                    entropy_evolution.append(0)
            
            ax3.plot(self.steps, entropy_evolution, marker='o', linewidth=2, markersize=6, 
                    label=f'Prompt {prompt_id}')
        
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Entropy')
        ax3.set_title('Prompt Entropy Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prompt similarity matrix (final step)
        ax4 = axes[1, 1]
        if self.steps:
            final_step = max(self.steps)
            if final_step in self.data:
                # Calculate pairwise similarities between prompts
                prompt_vectors = {}
                for prompt_id in all_prompts:
                    if prompt_id in self.data[final_step]['prompt_distributions']:
                        dist = self.data[final_step]['prompt_distributions'][prompt_id]
                        total = sum(dist.values())
                        if total > 0:
                            vector = [dist.get(str(i), 0) / total for i in range(len(self.meanings))]
                            prompt_vectors[prompt_id] = vector
                
                if len(prompt_vectors) > 1:
                    prompt_ids = sorted(prompt_vectors.keys(), key=int)
                    similarity_matrix = np.zeros((len(prompt_ids), len(prompt_ids)))
                    
                    for i, prompt1 in enumerate(prompt_ids):
                        for j, prompt2 in enumerate(prompt_ids):
                            if i == j:
                                similarity_matrix[i, j] = 1.0
                            else:
                                vec1 = np.array(prompt_vectors[prompt1])
                                vec2 = np.array(prompt_vectors[prompt2])
                                similarity = 1 - cosine(vec1, vec2)
                                similarity_matrix[i, j] = similarity
                    
                    im = ax4.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
                    ax4.set_xticks(range(len(prompt_ids)))
                    ax4.set_xticklabels([f'P{p}' for p in prompt_ids])
                    ax4.set_yticks(range(len(prompt_ids)))
                    ax4.set_yticklabels([f'P{p}' for p in prompt_ids])
                    ax4.set_title(f'Prompt Similarity Matrix (Step {final_step})')
                    
                    # Add text annotations
                    for i in range(len(prompt_ids)):
                        for j in range(len(prompt_ids)):
                            text = ax4.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                           ha="center", va="center", color="black", fontsize=8)
                    
                    plt.colorbar(im, ax=ax4, label='Cosine Similarity')
        
        plt.tight_layout()
        plt.savefig(output_path / 'prompt_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved prompt comparison analysis to {output_path / 'prompt_comparison_analysis.png'}")
    
    def create_prompt_meaning_evolution(self, output_dir: str = "visualizations"):
        """Create detailed evolution of each meaning for each prompt."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all prompt IDs
        all_prompts = set()
        for step_data in self.data.values():
            all_prompts.update(step_data['prompt_distributions'].keys())
        all_prompts = sorted(all_prompts, key=int)
        
        # Create subplots for each meaning
        n_meanings = len(self.meanings)
        n_cols = 2
        n_rows = (n_meanings + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for meaning_idx in range(n_meanings):
            row = meaning_idx // n_cols
            col = meaning_idx % n_cols
            ax = axes[row, col]
            
            # Plot evolution for each prompt
            for prompt_id in all_prompts:
                prompt_meaning_evolution = []
                for step in self.steps:
                    if step in self.data and prompt_id in self.data[step]['prompt_distributions']:
                        dist = self.data[step]['prompt_distributions'][prompt_id]
                        total = sum(dist.values())
                        if total > 0:
                            percentage = dist.get(str(meaning_idx), 0) / total * 100
                            prompt_meaning_evolution.append(percentage)
                        else:
                            prompt_meaning_evolution.append(0)
                    else:
                        prompt_meaning_evolution.append(0)
                
                ax.plot(self.steps, prompt_meaning_evolution, marker='o', linewidth=2, markersize=4, 
                       label=f'Prompt {prompt_id}')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Meaning {meaning_idx} Evolution by Prompt\n{self.meanings[meaning_idx][:40]}...')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for meaning_idx in range(n_meanings, n_rows * n_cols):
            row = meaning_idx // n_cols
            col = meaning_idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'prompt_meaning_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved prompt meaning evolution to {output_path / 'prompt_meaning_evolution.png'}")
    
    def create_prompt_convergence_analysis(self, output_dir: str = "visualizations"):
        """Analyze how prompts converge or diverge over time."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all prompt IDs
        all_prompts = set()
        for step_data in self.data.values():
            all_prompts.update(step_data['prompt_distributions'].keys())
        all_prompts = sorted(all_prompts, key=int)
        
        if len(all_prompts) < 2:
            print("Need at least 2 prompts for convergence analysis")
            return
        
        # Calculate pairwise distances between prompts over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average pairwise distance between prompts over time
        avg_distances = []
        for step in self.steps:
            if step in self.data:
                distances = []
                prompt_vectors = {}
                
                # Get vectors for all prompts at this step
                for prompt_id in all_prompts:
                    if prompt_id in self.data[step]['prompt_distributions']:
                        dist = self.data[step]['prompt_distributions'][prompt_id]
                        total = sum(dist.values())
                        if total > 0:
                            vector = [dist.get(str(i), 0) / total for i in range(len(self.meanings))]
                            prompt_vectors[prompt_id] = vector
                
                # Calculate pairwise distances
                prompt_ids = list(prompt_vectors.keys())
                for i in range(len(prompt_ids)):
                    for j in range(i+1, len(prompt_ids)):
                        vec1 = np.array(prompt_vectors[prompt_ids[i]])
                        vec2 = np.array(prompt_vectors[prompt_ids[j]])
                        distance = cosine(vec1, vec2)
                        distances.append(distance)
                
                if distances:
                    avg_distances.append(np.mean(distances))
                else:
                    avg_distances.append(0)
            else:
                avg_distances.append(0)
        
        ax1.plot(self.steps, avg_distances, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Average Cosine Distance')
        ax1.set_title('Prompt Convergence Analysis\n(Average Distance Between Prompts)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual pairwise distances
        ax2 = ax1.twinx()
        for i, prompt1 in enumerate(all_prompts):
            for j, prompt2 in enumerate(all_prompts):
                if i < j:  # Only plot each pair once
                    pair_distances = []
                    for step in self.steps:
                        if step in self.data:
                            if (prompt1 in self.data[step]['prompt_distributions'] and 
                                prompt2 in self.data[step]['prompt_distributions']):
                                dist1 = self.data[step]['prompt_distributions'][prompt1]
                                dist2 = self.data[step]['prompt_distributions'][prompt2]
                                total1 = sum(dist1.values())
                                total2 = sum(dist2.values())
                                if total1 > 0 and total2 > 0:
                                    vec1 = [dist1.get(str(k), 0) / total1 for k in range(len(self.meanings))]
                                    vec2 = [dist2.get(str(k), 0) / total2 for k in range(len(self.meanings))]
                                    distance = cosine(vec1, vec2)
                                    pair_distances.append(distance)
                                else:
                                    pair_distances.append(0)
                            else:
                                pair_distances.append(0)
                        else:
                            pair_distances.append(0)
                    
                    ax2.plot(self.steps, pair_distances, alpha=0.5, linewidth=1, 
                           label=f'P{prompt1}-P{prompt2}')
        
        ax2.set_ylabel('Pairwise Cosine Distance')
        ax2.legend(fontsize=8, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'prompt_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved prompt convergence analysis to {output_path / 'prompt_convergence_analysis.png'}")
    
    def create_similarity_analysis(self, output_dir: str = "visualizations"):
        """Analyze similarity between distributions across steps."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate distribution vectors for each step
        step_vectors = {}
        for step in self.steps:
            if step in self.data:
                overall_freq = self.data[step]['overall_frequencies']
                total = sum(overall_freq.values())
                # Convert string keys to int for proper indexing
                vector = [overall_freq.get(str(i), 0) / total for i in range(len(self.meanings))]
                step_vectors[step] = vector
        
        # Calculate pairwise similarities
        steps_list = sorted(step_vectors.keys())
        similarity_matrix = np.zeros((len(steps_list), len(steps_list)))
        
        for i, step1 in enumerate(steps_list):
            for j, step2 in enumerate(steps_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Use cosine similarity
                    vec1 = np.array(step_vectors[step1])
                    vec2 = np.array(step_vectors[step2])
                    similarity = 1 - cosine(vec1, vec2)
                    similarity_matrix[i, j] = similarity
        
        # Create similarity heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(steps_list)))
        ax.set_xticklabels(steps_list)
        ax.set_yticks(range(len(steps_list)))
        ax.set_yticklabels(steps_list)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Training Step')
        ax.set_title('Distribution Similarity Matrix (Cosine Similarity)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(steps_list)):
            for j in range(len(steps_list)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create evolution distance plot
        distances_from_initial = []
        initial_vector = np.array(step_vectors[steps_list[0]])
        
        for step in steps_list:
            current_vector = np.array(step_vectors[step])
            distance = cosine(initial_vector, current_vector)
            distances_from_initial.append(distance)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps_list, distances_from_initial, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Distance from Initial Distribution (Cosine Distance)', fontsize=12)
        ax.set_title('Evolution Distance from Initial Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'evolution_distance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved similarity analysis to {output_path}")
    
    def create_entropy_analysis(self, output_dir: str = "visualizations"):
        """Analyze entropy evolution to measure distribution diversity."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate entropy for each step
        step_entropies = {}
        prompt_entropies = {}
        
        for step in self.steps:
            if step in self.data:
                # Overall entropy
                overall_freq = self.data[step]['overall_frequencies']
                total = sum(overall_freq.values())
                if total > 0:
                    # Convert string keys to int for proper indexing
                    probs = [overall_freq.get(str(i), 0) / total for i in range(len(self.meanings))]
                    step_entropies[step] = entropy(probs)
                
                # Prompt-wise entropy
                prompt_entropies[step] = {}
                for prompt_id, dist in self.data[step]['prompt_distributions'].items():
                    total_prompt = sum(dist.values())
                    if total_prompt > 0:
                        # Convert string keys to int for proper indexing
                        probs = [dist.get(str(i), 0) / total_prompt for i in range(len(self.meanings))]
                        prompt_entropies[step][prompt_id] = entropy(probs)
        
        # Plot overall entropy evolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        steps_list = sorted(step_entropies.keys())
        entropies = [step_entropies[step] for step in steps_list]
        
        ax1.plot(steps_list, entropies, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Entropy', fontsize=12)
        ax1.set_title('Overall Distribution Entropy Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot prompt-wise entropy
        all_prompts = set()
        for step_data in prompt_entropies.values():
            all_prompts.update(step_data.keys())
        all_prompts = sorted(all_prompts, key=int)
        
        for prompt_id in all_prompts:
            prompt_entropy_values = []
            for step in steps_list:
                if step in prompt_entropies and prompt_id in prompt_entropies[step]:
                    prompt_entropy_values.append(prompt_entropies[step][prompt_id])
                else:
                    prompt_entropy_values.append(np.nan)
            
            ax2.plot(steps_list, prompt_entropy_values, marker='o', linewidth=2, markersize=6, 
                    label=f'Prompt {prompt_id}')
        
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Entropy', fontsize=12)
        ax2.set_title('Prompt-wise Distribution Entropy Evolution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'entropy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved entropy analysis to {output_path / 'entropy_evolution.png'}")
    
    def create_detailed_meaning_analysis(self, output_dir: str = "visualizations"):
        """Create detailed analysis for each meaning category."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subplots for each meaning
        n_meanings = len(self.meanings)
        n_cols = 2
        n_rows = (n_meanings + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for meaning_idx in range(n_meanings):
            row = meaning_idx // n_cols
            col = meaning_idx % n_cols
            ax = axes[row, col]
            
            # Get evolution data for this meaning
            meaning_evolution = []
            for step in self.steps:
                if step in self.data:
                    overall_freq = self.data[step]['overall_frequencies']
                    total = sum(overall_freq.values())
                    # Convert string keys to int for proper indexing
                    percentage = overall_freq.get(str(meaning_idx), 0) / total * 100
                    meaning_evolution.append(percentage)
                else:
                    meaning_evolution.append(0)
            
            # Plot evolution
            ax.plot(self.steps, meaning_evolution, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Training Step', fontsize=10)
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_title(f'Meaning {meaning_idx} Evolution\n{self.meanings[meaning_idx][:50]}...', 
                        fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(meaning_evolution) > 1:
                z = np.polyfit(self.steps, meaning_evolution, 1)
                p = np.poly1d(z)
                ax.plot(self.steps, p(self.steps), "--", alpha=0.8, color='red', 
                       label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
                ax.legend(fontsize=8)
        
        # Hide empty subplots
        for meaning_idx in range(n_meanings, n_rows * n_cols):
            row = meaning_idx // n_cols
            col = meaning_idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'detailed_meaning_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved detailed meaning analysis to {output_path / 'detailed_meaning_evolution.png'}")
    
    def create_summary_report(self, output_dir: str = "visualizations"):
        """Create a comprehensive summary report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate summary statistics
        summary_stats = {}
        
        for step in self.steps:
            if step in self.data:
                overall_freq = self.data[step]['overall_frequencies']
                total = sum(overall_freq.values())
                
                # Most common meaning
                most_common = max(overall_freq.items(), key=lambda x: x[1])
                
                # Distribution diversity (number of meanings with >5% usage)
                # Convert string keys to int for proper indexing
                percentages = [overall_freq.get(str(i), 0) / total * 100 for i in range(len(self.meanings))]
                diverse_meanings = sum(1 for p in percentages if p > 5)
                
                summary_stats[step] = {
                    'total_sentences': total,
                    'most_common_meaning': most_common[0],
                    'most_common_percentage': most_common[1] / total * 100,
                    'diverse_meanings': diverse_meanings,
                    'entropy': entropy(percentages) if total > 0 else 0
                }
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Total sentences processed
        steps_list = sorted(summary_stats.keys())
        total_sentences = [summary_stats[step]['total_sentences'] for step in steps_list]
        ax1.plot(steps_list, total_sentences, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Total Sentences')
        ax1.set_title('Total Sentences Processed')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Most common meaning percentage
        most_common_percentages = [summary_stats[step]['most_common_percentage'] for step in steps_list]
        ax2.plot(steps_list, most_common_percentages, marker='o', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Most Common Meaning Percentage')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of diverse meanings
        diverse_counts = [summary_stats[step]['diverse_meanings'] for step in steps_list]
        ax3.plot(steps_list, diverse_counts, marker='o', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Number of Meanings')
        ax3.set_title('Number of Meanings with >5% Usage')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Entropy evolution
        entropies = [summary_stats[step]['entropy'] for step in steps_list]
        ax4.plot(steps_list, entropies, marker='o', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Distribution Entropy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary data
        summary_data = {
            'steps': steps_list,
            'statistics': summary_stats,
            'meanings': self.meanings
        }
        
        with open(output_path / 'summary_statistics.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Saved summary report to {output_path}")
    
    def run_all_visualizations(self, output_dir: str = "visualizations"):
        """Run all visualization functions."""
        if not self.steps:
            print("No data available for visualization!")
            return
        
        if len(self.steps) < 2:
            print(f"Only {len(self.steps)} step(s) available. Need at least 2 steps for meaningful analysis.")
            return
        
        print(f"Creating comprehensive visualization suite for {len(self.steps)} steps...")
        print(f"Steps: {self.steps}")
        
        self.create_overall_distribution_evolution(output_dir)
        self.create_prompt_distribution_heatmaps(output_dir)
        self.create_prompt_comparison_analysis(output_dir)
        self.create_prompt_meaning_evolution(output_dir)
        self.create_prompt_convergence_analysis(output_dir)
        self.create_similarity_analysis(output_dir)
        self.create_entropy_analysis(output_dir)
        self.create_detailed_meaning_analysis(output_dir)
        self.create_summary_report(output_dir)
        
        print(f"\nAll visualizations saved to {output_dir}/")
        print("Generated files:")
        print("- meaning_distribution_evolution.png")
        print("- prompt_distribution_heatmaps.png")
        print("- prompt_comparison_analysis.png")
        print("- prompt_meaning_evolution.png")
        print("- prompt_convergence_analysis.png")
        print("- distribution_similarity_matrix.png")
        print("- evolution_distance.png")
        print("- entropy_evolution.png")
        print("- detailed_meaning_evolution.png")
        print("- summary_statistics.png")
        print("- summary_statistics.json")


def main():
    parser = argparse.ArgumentParser(description="Visualize classification evolution")
    parser.add_argument("--base-dir", 
                       default="../../verl_few_shot/Qwen2.5-Math-1.5B-dsr_sub/logic_sentences",
                       help="Base directory containing global_step_* folders")
    parser.add_argument("--output-dir", default="visualizations",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer and run analysis
    visualizer = ClassificationEvolutionVisualizer(args.base_dir)
    visualizer.load_data()
    visualizer.run_all_visualizations(args.output_dir)


if __name__ == "__main__":
    main() 