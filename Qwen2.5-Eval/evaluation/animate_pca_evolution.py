#!/usr/bin/env python3
"""
Animate PCA scatter plots showing how embedding distributions evolve across training steps.
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
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np


def _extract_step_from_path(path: str) -> Optional[int]:
    """Extract training step from file path."""
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


def _compute_pca_components(embeddings: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute PCA components for embeddings."""
    X = embeddings.to(torch.float32)
    # Center
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    # PCA via torch.pca_lowrank
    U, S, V = torch.pca_lowrank(Xc, q=2)
    comps = V[:, :2]
    return comps, mean


def _compute_ellipse_params(pts: Tensor) -> Tuple[List[float], float, float, float]:
    """Compute 95% confidence ellipse parameters."""
    if pts.shape[0] < 3:
        return [0.0, 0.0], 0.0, 0.0, 0.0
    
    mu = pts.mean(dim=0)
    C = torch.cov(pts.T)
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    angle = torch.atan2(evecs[1, 0], evecs[0, 0]).item()
    chi2_95 = 5.991
    width = 2 * (chi2_95 * evals[0]).sqrt().item()
    height = 2 * (chi2_95 * evals[1]).sqrt().item()
    return mu.tolist(), width, height, angle


def _draw_ellipse(ax, center: List[float], width: float, height: float, angle: float, edgecolor=None, alpha: float = 1.0):
    """Draw ellipse on matplotlib axis."""
    e = Ellipse(xy=center, width=width, height=height, angle=angle * 180.0 / 3.14159265,
                facecolor='none', edgecolor=edgecolor, lw=1.5, alpha=alpha)
    ax.add_patch(e)


def create_pca_animation(
    pt_pattern: str,
    output_path: str,
    fps: int = 2,
    dpi: int = 150,
    figsize: Tuple[int, int] = (10, 8),
    show_ellipses: bool = True,
    show_points: bool = True,
    point_alpha: float = 0.6,
    point_size: int = 8,
    ellipse_alpha: float = 0.8,
    title_template: str = "PCA Evolution - Step {step}",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    projection: str = "global",  # "global" or "per-step"
    axis_mode: str = "global",   # "global" or "per-step"
) -> None:
    """Create animated PCA scatter plot showing embedding evolution.
    projection: "global" uses PCA computed over all steps; "per-step" recomputes PCA each step.
    axis_mode: "global" fixes axes for all frames; "per-step" adapts axes per frame.
    """
    
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
    
    # Load first file to get prompt structure and compute global PCA
    first_embeddings, first_sentences_prompt, _ = _load_embeddings_from_step(files[0])
    prompt_ids, groups = _group_indices_by_prompt(first_sentences_prompt)
    G = len(groups)
    
    print(f"Found {G} prompt groups: {prompt_ids}")
    
    # Compute global PCA components from all embeddings to ensure consistent projection
    all_embeddings = []
    for pt_file in files:
        embeddings, _, _ = _load_embeddings_from_step(pt_file)
        all_embeddings.append(embeddings)
    
    global_embeddings = torch.cat(all_embeddings, dim=0)
    global_comps, global_mean = _compute_pca_components(global_embeddings)
    
    # Project all embeddings to 2D and group per step by prompt id
    projected_data: Dict[int, List[Tensor]] = {}
    per_step_axes: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for step, pt_file in zip(steps, files):
        embeddings, sentences_prompt, _ = _load_embeddings_from_step(pt_file)
        Xc = embeddings.to(torch.float32) - global_mean
        X2 = (Xc @ global_comps).cpu()
        
        # Group indices per step
        curr_prompt_ids, curr_groups = _group_indices_by_prompt(sentences_prompt)
        curr_pid_to_group = {pid: X2[idxs] for pid, idxs in zip(curr_prompt_ids, curr_groups)}
        
        # Build step groups in the canonical prompt_ids order; empty if missing
        step_groups = []
        for pid in prompt_ids:
            grp = curr_pid_to_group.get(pid, torch.empty((0, 2)))
            step_groups.append(grp)
        projected_data[step] = step_groups
        
        # Compute per-step axes if needed later
        xs, ys = [], []
        for grp in step_groups:
            if grp.numel() == 0:
                continue
            xs.extend(grp[:, 0].tolist())
            ys.extend(grp[:, 1].tolist())
        if xs and ys:
            x_margin = (max(xs) - min(xs)) * 0.1
            y_margin = (max(ys) - min(ys)) * 0.1
            per_step_axes[step] = ((min(xs) - x_margin, max(xs) + x_margin),
                                   (min(ys) - y_margin, max(ys) + y_margin))
        else:
            per_step_axes[step] = ((-1.0, 1.0), (-1.0, 1.0))
    
    # Determine global limits for consistent axes
    if xlim is None or ylim is None:
        all_x = []
        all_y = []
        for step_data in projected_data.values():
            for group_data in step_data:
                if group_data.numel() == 0:
                    continue
                all_x.extend(group_data[:, 0].tolist())
                all_y.extend(group_data[:, 1].tolist())
        
        x_margin = (max(all_x) - min(all_x)) * 0.1 if all_x else 1.0
        y_margin = (max(all_y) - min(all_y)) * 0.1 if all_y else 1.0
        global_xlim = ((min(all_x) - x_margin) if all_x else -1.0, (max(all_x) + x_margin) if all_x else 1.0)
        global_ylim = ((min(all_y) - y_margin) if all_y else -1.0, (max(all_y) + y_margin) if all_y else 1.0)
    else:
        global_xlim = xlim
        global_ylim = ylim
    
    # Create animation
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap('tab20', G)
    
    def animate(frame_idx):
        ax.clear()
        step = steps[frame_idx]
        step_data = projected_data[step]
        
        # Plot points and ellipses for each group
        for idx, (group_data, pid) in enumerate(zip(step_data, prompt_ids)):
            color = colors(idx)
            
            if show_points and group_data.numel() > 0:
                ax.scatter(group_data[:, 0], group_data[:, 1], 
                          s=point_size, alpha=point_alpha, 
                          label=f"Prompt {pid}", color=color)
            
            if show_ellipses and group_data.shape[0] >= 3:
                center, width, height, angle = _compute_ellipse_params(group_data)
                _draw_ellipse(ax, center, width, height, angle, 
                             edgecolor=color, alpha=ellipse_alpha)
        
        # Axes
        if axis_mode == "per-step":
            xlim_use, ylim_use = per_step_axes.get(step, global_xlim), per_step_axes.get(step, global_ylim)
            if isinstance(xlim_use, tuple) and isinstance(ylim_use, tuple) and isinstance(xlim_use[0], tuple):
                # Unpack when stored together
                xlim_use, ylim_use = xlim_use
        else:
            xlim_use, ylim_use = global_xlim, global_ylim
        ax.set_xlim(xlim_use)
        ax.set_ylim(ylim_use)
        
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title(title_template.format(step=step))
        ax.legend(markerscale=2, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(steps), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    # Save animation
    if output_path.endswith('.gif'):
        print(f"Saving GIF animation to: {output_path}")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    elif output_path.endswith('.mp4'):
        print(f"Saving MP4 animation to: {output_path}")
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
    else:
        # Default to GIF
        output_path = output_path + '.gif'
        print(f"Saving GIF animation to: {output_path}")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close()
    print(f"Animation saved successfully!")


def create_side_by_side_comparison(
    pt_pattern: str,
    output_path: str,
    comparison_steps: List[int],
    fps: int = 1,
    dpi: int = 150,
    figsize: Tuple[int, int] = (16, 8),
    show_ellipses: bool = True,
    show_points: bool = True,
    point_alpha: float = 0.6,
    point_size: int = 8,
    ellipse_alpha: float = 0.8,
) -> None:
    """Create side-by-side comparison of PCA plots at different steps."""
    
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
    
    # Filter to requested steps
    available_steps = set(steps)
    requested_steps = [s for s in comparison_steps if s in available_steps]
    if not requested_steps:
        raise ValueError(f"None of the requested steps {comparison_steps} found in available steps {steps}")
    
    print(f"Creating comparison for steps: {requested_steps}")
    
    # Load first file to get prompt structure
    first_embeddings, first_sentences_prompt, _ = _load_embeddings_from_step(files[0])
    prompt_ids, groups = _group_indices_by_prompt(first_sentences_prompt)
    G = len(groups)
    
    # Compute global PCA components
    all_embeddings = []
    for pt_file in files:
        embeddings, _, _ = _load_embeddings_from_step(pt_file)
        all_embeddings.append(embeddings)
    
    global_embeddings = torch.cat(all_embeddings, dim=0)
    global_comps, global_mean = _compute_pca_components(global_embeddings)
    
    # Create subplots
    n_steps = len(requested_steps)
    ncols = min(4, n_steps)
    nrows = (n_steps + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_steps == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    colors = plt.cm.get_cmap('tab20', G)
    
    # Determine global limits
    all_x = []
    all_y = []
    for step in requested_steps:
        step_idx = steps.index(step)
        pt_file = files[step_idx]
        embeddings, _, _ = _load_embeddings_from_step(pt_file)
        Xc = embeddings.to(torch.float32) - global_mean
        X2 = (Xc @ global_comps).cpu()
        for idxs in groups:
            group_data = X2[idxs]
            all_x.extend(group_data[:, 0].tolist())
            all_y.extend(group_data[:, 1].tolist())
    
    x_margin = (max(all_x) - min(all_x)) * 0.1
    y_margin = (max(all_y) - min(all_y)) * 0.1
    global_xlim = (min(all_x) - x_margin, max(all_x) + x_margin)
    global_ylim = (min(all_y) - y_margin, max(all_y) + y_margin)
    
    # Plot each step
    for idx, step in enumerate(requested_steps):
        ax = axes[idx]
        step_idx = steps.index(step)
        pt_file = files[step_idx]
        
        embeddings, _, _ = _load_embeddings_from_step(pt_file)
        Xc = embeddings.to(torch.float32) - global_mean
        X2 = (Xc @ global_comps).cpu()
        
        for group_idx, (idxs, pid) in enumerate(zip(groups, prompt_ids)):
            color = colors(group_idx)
            group_data = X2[idxs]
            
            if show_points:
                ax.scatter(group_data[:, 0], group_data[:, 1], 
                          s=point_size, alpha=point_alpha, 
                          label=f"Prompt {pid}", color=color)
            
            if show_ellipses and group_data.shape[0] >= 3:
                center, width, height, angle = _compute_ellipse_params(group_data)
                _draw_ellipse(ax, center, width, height, angle, 
                             edgecolor=color, alpha=ellipse_alpha)
        
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.set_title(f"Step {step}")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:  # Only show legend on first plot
            ax.legend(markerscale=2, fontsize=8, ncol=2)
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("PCA Evolution Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Side-by-side comparison saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create animated PCA plots showing embedding evolution")
    parser.add_argument("pt_pattern", help="Glob pattern for .pt files")
    parser.add_argument("--output", default="", help="Output path for animation (.gif or .mp4)")
    parser.add_argument("--mode", choices=["animation", "comparison"], default="animation", 
                       help="Create animation or side-by-side comparison")
    parser.add_argument("--comparison-steps", nargs="*", type=int, default=[0, 50, 100, 150, 200],
                       help="Steps to show in comparison mode")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for animation")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output")
    parser.add_argument("--figsize", nargs=2, type=int, default=[10, 8], help="Figure size (width height)")
    parser.add_argument("--no-ellipses", action="store_true", help="Hide confidence ellipses")
    parser.add_argument("--no-points", action="store_true", help="Hide scatter points")
    parser.add_argument("--point-alpha", type=float, default=0.6, help="Alpha for scatter points")
    parser.add_argument("--point-size", type=int, default=8, help="Size of scatter points")
    parser.add_argument("--ellipse-alpha", type=float, default=0.8, help="Alpha for ellipses")
    parser.add_argument("--projection", choices=["global", "per-step"], default="global",
                       help="PCA projection mode: global PCA or per-step PCA")
    parser.add_argument("--axis-mode", choices=["global", "per-step"], default="global",
                       help="Axis mode: fixed global axes or per-step adaptive axes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Auto-generate output path if not provided
    output_path = args.output
    if not output_path:
        dir_path = os.path.dirname(args.pt_pattern.split('*')[0])
        if args.mode == "animation":
            output_path = os.path.join(dir_path, "pca_animation.gif")
        else:
            output_path = os.path.join(dir_path, "pca_comparison.png")
    
    figsize = tuple(args.figsize)
    
    if args.mode == "animation":
        create_pca_animation(
            pt_pattern=args.pt_pattern,
            output_path=output_path,
            fps=args.fps,
            dpi=args.dpi,
            figsize=figsize,
            show_ellipses=not args.no_ellipses,
            show_points=not args.no_points,
            point_alpha=args.point_alpha,
            point_size=args.point_size,
            ellipse_alpha=args.ellipse_alpha,
            projection=args.projection,
            axis_mode=args.axis_mode,
        )
    else:
        create_side_by_side_comparison(
            pt_pattern=args.pt_pattern,
            output_path=output_path,
            comparison_steps=args.comparison_steps,
            fps=args.fps,
            dpi=args.dpi,
            figsize=figsize,
            show_ellipses=not args.no_ellipses,
            show_points=not args.no_points,
            point_alpha=args.point_alpha,
            point_size=args.point_size,
            ellipse_alpha=args.ellipse_alpha,
        )


if __name__ == "__main__":
    main() 