#!/usr/bin/env python3
"""
Analyze how prompt distribution similarities evolve across training steps.
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_step_from_path(path: str) -> Optional[int]:
    """Extract training step from file path."""
    # Look for patterns like "step_100", "global_step_100", "checkpoint_100"
    patterns = [
        r"step_(\d+)",
        r"global_step_(\d+)", 
        r"checkpoint_(\d+)",
        r"epoch_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def _load_embeddings_from_step(pt_path: str) -> Tuple[Tensor, List[str], int]:
    """Load embeddings and extract step number."""
    data = torch.load(pt_path, map_location="cpu")
    embeddings: Tensor = data["embeddings"]
    sentences_prompt: List[str] = data.get("sentences_prompt", [])
    
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings tensor in: {pt_path}")
    if not sentences_prompt or len(sentences_prompt) != embeddings.shape[0]:
        raise ValueError(f"'sentences_prompt' missing or length mismatch in: {pt_path}")
    
    step = _extract_step_from_path(pt_path)
    if step is None:
        raise ValueError(f"Could not extract step number from path: {pt_path}")
    
    return embeddings, sentences_prompt, step


def _group_indices_by_prompt(sentences_prompt: List[str]) -> Tuple[List[int], List[List[int]]]:
    """Group sentence indices by prompt ID."""
    prompt_to_indices: dict[int, List[int]] = {}
    for i, s in enumerate(sentences_prompt):
        m = re.match(r"Prompt\s+(\d+):", s)
        if not m:
            continue
        pid = int(m.group(1))
        prompt_to_indices.setdefault(pid, []).append(i)
    prompt_ids = sorted(prompt_to_indices.keys())
    groups = [prompt_to_indices[pid] for pid in prompt_ids]
    return prompt_ids, groups


def _compute_centroid_cosine_matrix(embeddings: Tensor, groups: List[List[int]]) -> Tensor:
    """Compute centroid cosine similarity matrix."""
    centroids: List[Tensor] = []
    for idxs in groups:
        group_embs = embeddings[idxs]
        centroid = group_embs.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        centroids.append(centroid.squeeze(0))
    C = torch.stack(centroids, dim=0)
    return (F.normalize(C, p=2, dim=1) @ F.normalize(C, p=2, dim=1).T).cpu()


def _compute_mmd_matrix(embeddings: Tensor, groups: List[List[int]], subsample: int = 2048) -> Tensor:
    """Compute MMD^2 matrix with RBF kernel."""
    def _pairwise_distances(A: Tensor, B: Tensor) -> Tensor:
        a2 = (A * A).sum(dim=1, keepdim=True)
        b2 = (B * B).sum(dim=1, keepdim=True).T
        ab = A @ B.T
        d2 = torch.clamp(a2 + b2 - 2 * ab, min=0.0)
        return torch.sqrt(d2 + 1e-12)
    
    def _subsample(X: Tensor, k: int) -> Tensor:
        if X.shape[0] <= k:
            return X
        idx = torch.randperm(X.shape[0])[:k]
        return X[idx]
    
    def _median_heuristic(X: Tensor, k: int = 2048) -> float:
        Xs = _subsample(X, k)
        d = _pairwise_distances(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
        vals = d[iu[0], iu[1]]
        med = torch.median(vals)
        sigma2 = (med.item() ** 2) if med.item() > 0 else 1.0
        return 1.0 / (2.0 * sigma2)
    
    def _rbf_kernel(X: Tensor, Y: Tensor, gamma: float) -> Tensor:
        d2 = _pairwise_distances(X, Y) ** 2
        return torch.exp(-gamma * d2)
    
    def _mmd2_unbiased(X: Tensor, Y: Tensor, gamma: float) -> Tensor:
        m, n = X.shape[0], Y.shape[0]
        Kxx = _rbf_kernel(X, X, gamma)
        Kyy = _rbf_kernel(Y, Y, gamma)
        Kxy = _rbf_kernel(X, Y, gamma)
        sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
        sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
        sum_Kxy = Kxy.mean()
        return sum_Kxx + sum_Kyy - 2 * sum_Kxy
    
    # Use global median heuristic
    Xall = _subsample(embeddings.to(torch.float32), subsample)
    gamma = _median_heuristic(Xall, k=min(subsample, 4096))
    
    G = len(groups)
    M = torch.zeros((G, G), dtype=torch.float64)
    cache: Dict[int, Tensor] = {}
    
    for i in range(G):
        Xi = cache.get(i)
        if Xi is None:
            Xi = _subsample(embeddings[groups[i]].to(torch.float32), subsample)
            cache[i] = Xi
        for j in range(G):
            Xj = cache.get(j)
            if Xj is None:
                Xj = _subsample(embeddings[groups[j]].to(torch.float32), subsample)
                cache[j] = Xj
            mmd2 = _mmd2_unbiased(Xi, Xj, gamma)
            M[i, j] = mmd2.to(torch.float64)
    
    return M.to(torch.float32)


def analyze_temporal_similarities(
    pt_pattern: str,
    metric: str = "centroid_cosine",
    save_prefix: Optional[str] = None,
    mmd_subsample: int = 2048,
    plot_every_n_steps: int = 1,
) -> None:
    """Analyze how prompt distribution similarities change across training steps."""
    
    # Find all matching .pt files
    pt_files = glob.glob(pt_pattern)
    if not pt_files:
        raise ValueError(f"No files found matching pattern: {pt_pattern}")
    
    # Sort by step number
    step_file_pairs = []
    for pt_file in pt_files:
        step = _extract_step_from_path(pt_file)
        if step is not None:
            step_file_pairs.append((step, pt_file))
    
    step_file_pairs.sort(key=lambda x: x[0])
    steps = [pair[0] for pair in step_file_pairs]
    files = [pair[1] for pair in step_file_pairs]
    
    print(f"Found {len(files)} checkpoint files from steps: {steps}")
    
    # Load first file to get prompt structure
    first_embeddings, first_sentences_prompt, _ = _load_embeddings_from_step(files[0])
    prompt_ids, groups = _group_indices_by_prompt(first_sentences_prompt)
    G = len(groups)
    
    print(f"Found {G} prompt groups: {prompt_ids}")
    
    # Compute similarity matrices for each step
    similarity_matrices: Dict[int, Tensor] = {}
    
    for step, pt_file in zip(steps, files):
        print(f"Processing step {step}...")
        embeddings, sentences_prompt, _ = _load_embeddings_from_step(pt_file)
        
        # Verify prompt structure is consistent
        curr_prompt_ids, curr_groups = _group_indices_by_prompt(sentences_prompt)
        if curr_prompt_ids != prompt_ids:
            print(f"Warning: prompt IDs changed at step {step}: {curr_prompt_ids} vs {prompt_ids}")
            continue
        
        if metric == "centroid_cosine":
            sim_mat = _compute_centroid_cosine_matrix(embeddings, curr_groups)
        elif metric == "mmd":
            sim_mat = _compute_mmd_matrix(embeddings, curr_groups, subsample=mmd_subsample)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        similarity_matrices[step] = sim_mat
    
    if not similarity_matrices:
        raise ValueError("No valid similarity matrices computed")
    
    # Create temporal plots
    if save_prefix:
        _plot_temporal_evolution(similarity_matrices, prompt_ids, save_prefix, metric, plot_every_n_steps)
        _plot_convergence_analysis(similarity_matrices, prompt_ids, save_prefix, metric)
        _plot_pairwise_evolution(similarity_matrices, prompt_ids, save_prefix, metric)


def _plot_temporal_evolution(
    similarity_matrices: Dict[int, Tensor],
    prompt_ids: List[int],
    save_prefix: str,
    metric: str,
    plot_every_n_steps: int,
) -> None:
    """Plot how each similarity value evolves over time."""
    steps = sorted(similarity_matrices.keys())
    G = len(prompt_ids)
    
    # Create subplots for each pair
    n_pairs = G * (G - 1) // 2
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    
    plt.figure(figsize=(4 * ncols, 3 * nrows))
    
    pair_idx = 0
    for i in range(G):
        for j in range(i + 1, G):
            plt.subplot(nrows, ncols, pair_idx + 1)
            
            # Extract similarity values for this pair across all steps
            sim_vals = []
            for step in steps:
                if step % plot_every_n_steps == 0:  # Plot every nth step
                    sim_val = similarity_matrices[step][i, j].item()
                    sim_vals.append(sim_val)
            
            plot_steps = [s for s in steps if s % plot_every_n_steps == 0]
            plt.plot(plot_steps, sim_vals, 'o-', markersize=3, linewidth=1)
            plt.title(f"Prompt {prompt_ids[i]} vs {prompt_ids[j]}")
            plt.xlabel("Training Step")
            plt.ylabel(f"{metric} similarity")
            plt.grid(True, alpha=0.3)
            
            pair_idx += 1
    
    plt.suptitle(f"Temporal evolution of {metric} similarities")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{save_prefix}_temporal_evolution_{metric}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal evolution plot: {save_prefix}_temporal_evolution_{metric}.png")


def _plot_convergence_analysis(
    similarity_matrices: Dict[int, Tensor],
    prompt_ids: List[int],
    save_prefix: str,
    metric: str,
) -> None:
    """Plot convergence analysis: variance over time and final vs initial similarities."""
    steps = sorted(similarity_matrices.keys())
    G = len(prompt_ids)
    
    # Compute variance of each similarity value over time
    all_similarities = torch.stack([similarity_matrices[step] for step in steps], dim=0)
    variances = torch.var(all_similarities, dim=0)
    
    # Plot variance heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(variances.cpu(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Variance over time")
    plt.xticks(range(len(prompt_ids)), prompt_ids, rotation=45, ha="right")
    plt.yticks(range(len(prompt_ids)), prompt_ids)
    plt.title(f"Variance of {metric} similarities across training")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_variance_heatmap_{metric}.png", dpi=200)
    plt.close()
    
    # Plot initial vs final similarities
    initial_sim = similarity_matrices[steps[0]]
    final_sim = similarity_matrices[steps[-1]]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(initial_sim.flatten().cpu(), final_sim.flatten().cpu(), alpha=0.6)
    
    # Add diagonal line
    min_val = min(initial_sim.min().item(), final_sim.min().item())
    max_val = max(initial_sim.max().item(), final_sim.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='No change')
    
    plt.xlabel(f"Initial {metric} similarity")
    plt.ylabel(f"Final {metric} similarity")
    plt.title(f"Initial vs Final {metric} similarities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_initial_vs_final_{metric}.png", dpi=200)
    plt.close()
    
    print(f"Saved convergence analysis plots for {metric}")


def _plot_pairwise_evolution(
    similarity_matrices: Dict[int, Tensor],
    prompt_ids: List[int],
    save_prefix: str,
    metric: str,
) -> None:
    """Plot evolution of specific pairwise similarities."""
    steps = sorted(similarity_matrices.keys())
    G = len(prompt_ids)
    
    # Find the most and least similar pairs in the final step
    final_sim = similarity_matrices[steps[-1]]
    
    # Get all pairs (excluding diagonal)
    pairs = []
    for i in range(G):
        for j in range(i + 1, G):
            pairs.append((i, j, final_sim[i, j].item()))
    
    # Sort by final similarity
    pairs.sort(key=lambda x: x[2])
    
    # Plot top 3 most similar and bottom 3 least similar pairs
    plt.figure(figsize=(12, 8))
    
    # Most similar pairs
    plt.subplot(2, 1, 1)
    for i, j, final_val in pairs[-3:]:
        sim_vals = [similarity_matrices[step][i, j].item() for step in steps]
        plt.plot(steps, sim_vals, 'o-', label=f"Prompt {prompt_ids[i]} vs {prompt_ids[j]} (final: {final_val:.3f})")
    plt.title(f"Most similar pairs - {metric}")
    plt.ylabel(f"{metric} similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Least similar pairs
    plt.subplot(2, 1, 2)
    for i, j, final_val in pairs[:3]:
        sim_vals = [similarity_matrices[step][i, j].item() for step in steps]
        plt.plot(steps, sim_vals, 'o-', label=f"Prompt {prompt_ids[i]} vs {prompt_ids[j]} (final: {final_val:.3f})")
    plt.title(f"Least similar pairs - {metric}")
    plt.xlabel("Training Step")
    plt.ylabel(f"{metric} similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_pairwise_evolution_{metric}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved pairwise evolution plot: {save_prefix}_pairwise_evolution_{metric}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal evolution of prompt distribution similarities")
    parser.add_argument("pt_pattern", help="Glob pattern for .pt files (e.g., 'path/to/step_*/embeddings.pt')")
    parser.add_argument("--metric", choices=["centroid_cosine", "mmd"], default="centroid_cosine", help="Similarity metric to use")
    parser.add_argument("--save-prefix", default="", help="Prefix for saving plots")
    parser.add_argument("--mmd-subsample", type=int, default=2048, help="Subsample size for MMD computation")
    parser.add_argument("--plot-every-n-steps", type=int, default=1, help="Plot every nth step to reduce clutter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Auto-generate save prefix if not provided
    save_prefix = args.save_prefix
    if not save_prefix:
        # Extract directory from pattern
        dir_path = os.path.dirname(args.pt_pattern.split('*')[0])
        save_prefix = os.path.join(dir_path, "temporal_analysis")
    
    analyze_temporal_similarities(
        pt_pattern=args.pt_pattern,
        metric=args.metric,
        save_prefix=save_prefix,
        mmd_subsample=args.mmd_subsample,
        plot_every_n_steps=args.plot_every_n_steps,
    )


if __name__ == "__main__":
    main() 