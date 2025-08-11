#!/usr/bin/env python3
"""
Analyze similarities between prompt distributions from saved embeddings.
"""

import argparse
import csv
import re
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_cpu_float(t: Tensor) -> Tensor:
    return t.detach().to(torch.float32).cpu()


def _group_indices_by_prompt(sentences_prompt: List[str]) -> Tuple[List[int], List[List[int]]]:
    """Group sentence indices by prompt ID."""
    prompt_to_indices: dict[int, List[int]] = {}
    for i, s in enumerate(sentences_prompt):
        m = re.match(r"Prompt\s+(\d+):", s)
        if not m:
            # Skip if it doesn't match the expected format
            continue
        pid = int(m.group(1))
        prompt_to_indices.setdefault(pid, []).append(i)
    prompt_ids = sorted(prompt_to_indices.keys())
    groups = [prompt_to_indices[pid] for pid in prompt_ids]
    return prompt_ids, groups


def analyze_prompt_distributions(
    pt_path: str,
    metric: str = "centroid_cosine",
    save_csv_path: Optional[str] = None,
    top_k: int = 3,
    save_prefix: Optional[str] = None,
    mmd_subsample: int = 2048,
    shape_metrics: Optional[List[str]] = None,
    pair_subsample_pairs: int = 50000,
    mds_from: str = "mmd",  # which distance matrix to embed: mmd|energy|gaussian_w2|pair_dist|pair_angle|cov_shape
    qq_from: str = "pair_dist",  # which distribution to use for QQ grid: pair_dist|pair_angle
    qq_top_pairs: int = 6,
) -> None:
    """Analyze similarities between prompt distributions using centroid cosine similarity."""
    print(f"Loading embeddings from: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    
    embeddings: Tensor = data["embeddings"]
    sentences_prompt: List[str] = data.get("sentences_prompt", [])
    
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Invalid embeddings tensor in the .pt file")
    if not sentences_prompt or len(sentences_prompt) != embeddings.shape[0]:
        raise ValueError("'sentences_prompt' missing or length mismatch with embeddings")

    prompt_ids, groups = _group_indices_by_prompt(sentences_prompt)
    if len(groups) < 2:
        print("Not enough prompt groups to compare")
        return

    print(f"\nFound {len(groups)} prompt groups:")
    for pid, idxs in zip(prompt_ids, groups):
        print(f"  Prompt {pid}: {len(idxs)} sentences")

    # Compute group centroids
    centroids: List[Tensor] = []
    group_sizes: List[int] = []
    for idxs in groups:
        group_embs = embeddings[idxs]
        centroid = group_embs.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        centroids.append(centroid.squeeze(0))
        group_sizes.append(group_embs.shape[0])
    C = torch.stack(centroids, dim=0)  # [G, D]

    # Similarity matrix based on centroids
    sim_mat = (F.normalize(C, p=2, dim=1) @ F.normalize(C, p=2, dim=1).T).cpu()

    print(f"\nCentroid cosine similarity matrix:")
    print("=" * 50)
    
    # Print similarity matrix
    header = "Prompt" + "".join([f"{pid:>8}" for pid in prompt_ids])
    print(header)
    print("-" * len(header))
    
    for i, pid in enumerate(prompt_ids):
        row = f"{pid:>5}"
        for j in range(len(prompt_ids)):
            sim_val = sim_mat[i, j].item()
            row += f"{sim_val:>8.4f}"
        print(row)

    print("\nNearest neighbors for each prompt:")
    print("=" * 50)
    
    # For each group, show top-k most similar other groups
    for i, pid in enumerate(prompt_ids):
        sims = sim_mat[i].clone()
        sims[i] = -1.0  # exclude self
        topk_vals, topk_idx = torch.topk(sims, k=min(top_k, len(prompt_ids) - 1))
        neighbors = ", ".join([f"Prompt {prompt_ids[j]} (cos={v:.4f})" for v, j in zip(topk_vals.tolist(), topk_idx.tolist())])
        print(f"Nearest to Prompt {pid}: {neighbors}")

    # Optionally save CSV of similarities
    if save_csv_path:
        import os
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["prompt_id"] + [str(pid) for pid in prompt_ids]
            writer.writerow(header)
            for i, pid in enumerate(prompt_ids):
                row = [str(pid)] + [f"{sim_mat[i, j]:.6f}" for j in range(len(prompt_ids))]
                writer.writerow(row)
        print(f"\nSaved centroid cosine similarity matrix CSV to: {save_csv_path}")

    # Save heatmap visualization of centroid cosine similarities
    if save_prefix:
        _plot_heatmap(sim_mat, prompt_ids, f"{save_prefix}_centroid_cosine.png", title="Centroid Cosine Similarity")

    # Distribution-level distances
    # 1) Gaussian 2-Wasserstein (Fréchet) distance between fitted Gaussians
    W2 = _gaussian_wasserstein2_matrix(embeddings, groups)
    if save_prefix:
        _plot_heatmap(W2, prompt_ids, f"{save_prefix}_gaussian_w2.png", title="Gaussian W2 Distance (lower is closer)")
    _maybe_print_topk("Gaussian W2", W2, prompt_ids, top_k)

    # 2) Energy distance
    ED = _energy_distance_matrix(embeddings, groups, subsample=mmd_subsample)
    if save_prefix:
        _plot_heatmap(ED, prompt_ids, f"{save_prefix}_energy_distance.png", title="Energy Distance (lower is closer)")
    _maybe_print_topk("Energy Distance", ED, prompt_ids, top_k)

    # 3) MMD^2 with RBF kernel
    MMD = _mmd_matrix(embeddings, groups, subsample=mmd_subsample)
    if save_prefix:
        _plot_heatmap(MMD, prompt_ids, f"{save_prefix}_mmd2.png", title="MMD^2 (RBF, lower is closer)")
    _maybe_print_topk("MMD^2", MMD, prompt_ids, top_k)

    # 2D PCA scatter of all embeddings colored by prompt + 95% covariance ellipses
    if save_prefix:
        _plot_pca_scatter_with_ellipses(embeddings, groups, prompt_ids, f"{save_prefix}_pca_scatter_ellipses.png")

    # Shape-only metrics (mean/scale-invariant)
    shape_metrics = shape_metrics or ["cov_shape", "pair_dist", "pair_angle"]

    Cshape_loge: Optional[Tensor] = None
    if "cov_shape" in shape_metrics:
        Cshape_loge = _cov_shape_distance_matrix(embeddings, groups, metric="loge")
        if save_prefix:
            _plot_heatmap(Cshape_loge, prompt_ids, f"{save_prefix}_covshape_loge.png", title="Covariance Shape Distance (log-Euclidean)")
        _maybe_print_topk("Covariance Shape (log-Euclidean)", Cshape_loge, prompt_ids, top_k)

    # Precompute pairwise distributions if requested
    pair_dist_list: Optional[List[Tensor]] = None
    pair_ang_list: Optional[List[Tensor]] = None

    W1d: Optional[Tensor] = None
    if "pair_dist" in shape_metrics:
        pair_dist_list = _pairwise_distance_distributions(embeddings, groups, num_pairs=pair_subsample_pairs, normalize_scale=True)
        W1d = _w1_from_sorted_distributions(pair_dist_list)
        if save_prefix:
            _plot_heatmap(W1d, prompt_ids, f"{save_prefix}_pairdist_w1.png", title="Pairwise Distance Distribution W1 (lower is closer)")
        _maybe_print_topk("Pairwise Distance W1", W1d, prompt_ids, top_k)

    W1a: Optional[Tensor] = None
    if "pair_angle" in shape_metrics:
        pair_ang_list = _pairwise_angle_distributions(embeddings, groups, num_pairs=pair_subsample_pairs)
        W1a = _w1_from_sorted_distributions(pair_ang_list)
        if save_prefix:
            _plot_heatmap(W1a, prompt_ids, f"{save_prefix}_pairangle_w1.png", title="Pairwise Angle Distribution W1 (lower is closer)")
        _maybe_print_topk("Pairwise Angle W1", W1a, prompt_ids, top_k)

    # Classical MDS map of groups using a chosen distance matrix
    if save_prefix:
        dist_map = {
            "mmd": MMD,
            "energy": ED,
            "gaussian_w2": W2,
            "pair_dist": W1d,
            "pair_angle": W1a,
            "cov_shape": Cshape_loge,
        }
        Dmat = dist_map.get(mds_from)
        if Dmat is not None:
            _plot_mds(Dmat, prompt_ids, f"{save_prefix}_mds_{mds_from}.png", title=f"MDS of group distances: {mds_from}")

    # QQ plot grid for closest pairs by a chosen distribution
    if save_prefix:
        if qq_from == "pair_dist" and pair_dist_list is not None and W1d is not None:
            pairs = _nearest_pairs_from_matrix(W1d, top_pairs=qq_top_pairs)
            _plot_qq_grid(pair_dist_list, prompt_ids, pairs, f"{save_prefix}_qq_pairdist.png", xlabel="Group A distances", ylabel="Group B distances", title="QQ plots of pairwise distance distributions")
        elif qq_from == "pair_angle" and pair_ang_list is not None and W1a is not None:
            pairs = _nearest_pairs_from_matrix(W1a, top_pairs=qq_top_pairs)
            _plot_qq_grid(pair_ang_list, prompt_ids, pairs, f"{save_prefix}_qq_pairangle.png", xlabel="Group A angles", ylabel="Group B angles", title="QQ plots of pairwise angle distributions")


def _maybe_print_topk(name: str, mat: Tensor, prompt_ids: List[int], top_k: int) -> None:
    print(f"\nNearest neighbors by {name}:")
    print("=" * 50)
    # If this is a distance matrix (lower better), we invert ordering
    for i, pid in enumerate(prompt_ids):
        vals = mat[i].clone()
        vals[i] = float("inf")
        k = min(top_k, len(prompt_ids) - 1)
        topk_vals, topk_idx = torch.topk(-vals, k=k)  # negative for ascending
        neighbors = ", ".join([f"Prompt {prompt_ids[j]} ({-v:.4f})" for v, j in zip(topk_vals.tolist(), topk_idx.tolist())])
        print(f"Nearest to Prompt {pid}: {neighbors}")


def _plot_heatmap(matrix: Tensor, labels: List[int], out_path: str, title: str = "") -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(_to_cpu_float(matrix), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def _plot_pca_scatter_with_ellipses(embeddings: Tensor, groups: List[List[int]], prompt_ids: List[int], out_path: str) -> None:
    X = embeddings.to(torch.float32)
    # Center
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    # PCA via torch.pca_lowrank
    U, S, V = torch.pca_lowrank(Xc, q=2)
    comps = V[:, :2]
    X2 = (Xc @ comps).cpu()

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab20', len(groups))
    for idx, (g, pid) in enumerate(zip(groups, prompt_ids)):
        pts = X2[g]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4, label=f"Prompt {pid}", color=colors(idx))
        # 95% covariance ellipse (chi2 2 dof quantile ~ 5.991)
        if pts.shape[0] >= 3:
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
            _draw_ellipse(mu.tolist(), width, height, angle, edgecolor=colors(idx), alpha=0.9)
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.title("PCA scatter with 95% covariance ellipses")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved PCA scatter with ellipses: {out_path}")


def _draw_ellipse(center: List[float], width: float, height: float, angle: float, edgecolor=None, alpha: float = 1.0):
    from matplotlib.patches import Ellipse
    e = Ellipse(xy=center, width=width, height=height, angle=angle * 180.0 / 3.14159265,
                facecolor='none', edgecolor=edgecolor, lw=1.5, alpha=alpha)
    ax = plt.gca()
    ax.add_patch(e)


def _safe_cov(X: Tensor, eps: float = 1e-6) -> Tensor:
    X = X.to(torch.float64)
    Xc = X - X.mean(dim=0, keepdim=True)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    # Regularize for numerical stability
    cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return cov


def _sqrt_spd(A: Tensor) -> Tensor:
    # A should be symmetric positive-definite
    # Compute eigen decomposition
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=0.0)
    sqrt_evals = torch.sqrt(evals)
    return (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T


def _gaussian_wasserstein2_matrix(embeddings: Tensor, groups: List[List[int]]) -> Tensor:
    # Compute W2 distance between Gaussians fitted to each group
    means: List[Tensor] = []
    covs: List[Tensor] = []
    for idxs in groups:
        X = embeddings[idxs]
        means.append(X.mean(dim=0))
        covs.append(_safe_cov(X))

    G = len(groups)
    W2 = torch.zeros((G, G), dtype=torch.float64)
    for i in range(G):
        for j in range(G):
            dm = torch.norm(means[i] - means[j]) ** 2
            # Compute sqrt(Ci^{1/2} Cj Ci^{1/2}) via congruence transform
            Ci_sqrt = _sqrt_spd(covs[i])
            middle = Ci_sqrt @ covs[j] @ Ci_sqrt
            sqrt_middle = _sqrt_spd(middle)
            term = torch.trace(covs[i] + covs[j] - 2.0 * sqrt_middle)
            W2[i, j] = (dm + term).to(torch.float64)
    return W2.to(torch.float32)


def _pairwise_distances(A: Tensor, B: Tensor) -> Tensor:
    # A: [m, d], B: [n, d]
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


def _energy_distance_matrix(embeddings: Tensor, groups: List[List[int]], subsample: int = 2048) -> Tensor:
    G = len(groups)
    ED = torch.zeros((G, G), dtype=torch.float64)
    for i in range(G):
        Xi = embeddings[groups[i]].to(torch.float32)
        Xi = _subsample(Xi, subsample)
        for j in range(G):
            Xj = embeddings[groups[j]].to(torch.float32)
            Xj = _subsample(Xj, subsample)
            # 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
            d_xy = _pairwise_distances(Xi, Xj).mean()
            d_xx = _pairwise_distances(Xi, Xi).mean()
            d_yy = _pairwise_distances(Xj, Xj).mean()
            ED[i, j] = (2 * d_xy - d_xx - d_yy).to(torch.float64)
    return ED.to(torch.float32)


def _rbf_kernel(X: Tensor, Y: Tensor, gammas: List[float]) -> Tensor:
    # gammas are 1/(2*sigma^2)
    d2 = _pairwise_distances(X, Y) ** 2
    K = sum(torch.exp(-g * d2) for g in gammas) / len(gammas)
    return K


def _median_heuristic(X: Tensor, k: int = 2048) -> float:
    Xs = _subsample(X, k)
    d = _pairwise_distances(Xs, Xs)
    # upper triangular without diagonal
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    vals = d[iu[0], iu[1]]
    med = torch.median(vals)
    sigma2 = (med.item() ** 2) if med.item() > 0 else 1.0
    gamma = 1.0 / (2.0 * sigma2)
    return gamma


def _mmd2_unbiased(X: Tensor, Y: Tensor, gammas: List[float]) -> Tensor:
    m = X.shape[0]
    n = Y.shape[0]
    Kxx = _rbf_kernel(X, X, gammas)
    Kyy = _rbf_kernel(Y, Y, gammas)
    Kxy = _rbf_kernel(X, Y, gammas)
    # Unbiased estimator: exclude diagonals
    sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    sum_Kxy = Kxy.mean()
    return sum_Kxx + sum_Kyy - 2 * sum_Kxy


def _mmd_matrix(embeddings: Tensor, groups: List[List[int]], subsample: int = 2048) -> Tensor:
    # Build a global gamma list using median heuristic on all embeddings
    Xall = _subsample(embeddings.to(torch.float32), subsample)
    gamma = _median_heuristic(Xall, k=min(subsample, 4096))
    gammas = [gamma / f for f in [4.0, 2.0, 1.0, 0.5]]

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
            mmd2 = _mmd2_unbiased(Xi, Xj, gammas)
            M[i, j] = mmd2.to(torch.float64)
    return M.to(torch.float32)


def _cov_shape_distance_matrix(embeddings: Tensor, groups: List[List[int]], metric: str = "loge") -> Tensor:
    """Compare covariance shape matrices (translation removed via centering, scale removed via trace normalization).

    metric:
      - 'loge': log-Euclidean distance ||log(Si) - log(Sj)||_F
    """
    shapes: List[Tensor] = []
    for idxs in groups:
        X = embeddings[idxs].to(torch.float64)
        C = _safe_cov(X)
        C = C / torch.trace(C)  # remove scale
        shapes.append(C)
    G = len(shapes)
    D = torch.zeros((G, G), dtype=torch.float64)
    if metric == "loge":
        # compute log-matrices via eigendecomposition
        logmats: List[Tensor] = []
        for S in shapes:
            evals, evecs = torch.linalg.eigh(S)
            L = (evecs * torch.log(torch.clamp(evals, min=1e-12)).unsqueeze(0)) @ evecs.T
            logmats.append(L)
        for i in range(G):
            for j in range(G):
                diff = logmats[i] - logmats[j]
                D[i, j] = torch.norm(diff, p="fro")
    return D.to(torch.float32)


def _pairwise_distance_distributions(embeddings: Tensor, groups: List[List[int]], num_pairs: int = 50000, normalize_scale: bool = True) -> List[Tensor]:
    dists: List[Tensor] = []
    for idxs in groups:
        X = embeddings[idxs].to(torch.float32)
        n = X.shape[0]
        if n < 2:
            dists.append(torch.zeros(1))
            continue
        i1, i2 = _sample_pair_indices(n, num_pairs)
        d = torch.norm(X[i1] - X[i2], dim=1)
        if normalize_scale:
            med = d.median()
            if med > 0:
                d = d / med
        dists.append(torch.sort(d).values)
    return dists


def _pairwise_angle_distributions(embeddings: Tensor, groups: List[List[int]], num_pairs: int = 50000) -> List[Tensor]:
    angles: List[Tensor] = []
    for idxs in groups:
        X = embeddings[idxs].to(torch.float32)
        n = X.shape[0]
        if n < 2:
            angles.append(torch.zeros(1))
            continue
        i1, i2 = _sample_pair_indices(n, num_pairs)
        x1 = F.normalize(X[i1], p=2, dim=1)
        x2 = F.normalize(X[i2], p=2, dim=1)
        cos = (x1 * x2).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.arccos(cos)
        angles.append(torch.sort(ang).values)
    return angles


def _w1_from_sorted_distributions(sorted_lists: List[Tensor]) -> Tensor:
    G = len(sorted_lists)
    W = torch.zeros((G, G), dtype=torch.float64)
    for i in range(G):
        for j in range(G):
            m = min(sorted_lists[i].numel(), sorted_lists[j].numel())
            if m == 0:
                W[i, j] = 0.0
            else:
                a = sorted_lists[i][:m]
                b = sorted_lists[j][:m]
                W[i, j] = torch.mean(torch.abs(a - b)).to(torch.float64)
    return W.to(torch.float32)


def _sample_pair_indices(n: int, num_pairs: int) -> Tuple[Tensor, Tensor]:
    max_pairs = n * (n - 1) // 2
    k = min(num_pairs, max_pairs)
    idx1 = torch.randint(0, n, (k,))
    idx2 = torch.randint(0, n, (k,))
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    if idx1.numel() == 0:
        return torch.tensor([0]), torch.tensor([0])
    return idx1, idx2


def _nearest_pairs_from_matrix(D: Tensor, top_pairs: int = 6) -> List[Tuple[int, int]]:
    # Return list of (i,j) with i<j of smallest distances
    G = D.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(G):
        for j in range(i + 1, G):
            pairs.append((i, j, float(D[i, j])))
    pairs.sort(key=lambda x: x[2])
    return [(i, j) for i, j, _ in pairs[:top_pairs]]


def _plot_qq_grid(sorted_lists: List[Tensor], prompt_ids: List[int], pairs: List[Tuple[int, int]], out_path: str, xlabel: str, ylabel: str, title: str) -> None:
    if not pairs:
        return
    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    plt.figure(figsize=(4 * ncols, 3.2 * nrows))
    for k, (i, j) in enumerate(pairs):
        plt.subplot(nrows, ncols, k + 1)
        a = sorted_lists[i]
        b = sorted_lists[j]
        m = min(a.numel(), b.numel())
        if m == 0:
            plt.axis('off')
            continue
        aa = a[:m].cpu().numpy()
        bb = b[:m].cpu().numpy()
        plt.plot(aa, bb, '.', ms=2, alpha=0.6)
        lim = [min(aa.min(), bb.min()), max(aa.max(), bb.max())]
        plt.plot(lim, lim, 'k--', lw=1)
        plt.title(f"Prompt {prompt_ids[i]} vs {prompt_ids[j]}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved QQ grid: {out_path}")


def _plot_mds(D: Tensor, prompt_ids: List[int], out_path: str, title: str = "MDS map") -> None:
    # Classical MDS from distance matrix
    D2 = (D ** 2).to(torch.float64)
    n = D2.shape[0]
    J = torch.eye(n, dtype=torch.float64) - torch.ones((n, n), dtype=torch.float64) / n
    B = -0.5 * J @ D2 @ J
    evals, evecs = torch.linalg.eigh(B)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals.clamp(min=0)
    # take top-2 positive components
    k = min(2, (pos > 0).sum().item())
    if k == 0:
        print("MDS: no positive eigenvalues; cannot embed")
        return
    coords = evecs[:, :k] * torch.sqrt(pos[:k]).unsqueeze(0)
    X = coords.to(torch.float32).cpu()
    plt.figure(figsize=(6, 5))
    colors = plt.cm.get_cmap('tab20', len(prompt_ids))
    for i, pid in enumerate(prompt_ids):
        plt.scatter(X[i, 0].item(), X[i, 1].item(), color=colors(i), s=60)
        plt.text(X[i, 0].item() + 0.02, X[i, 1].item() + 0.02, str(pid), fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved MDS map: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze similarities between prompt distributions from saved embeddings")
    parser.add_argument("pt_path", help="Path to saved embeddings .pt file")
    parser.add_argument("--save-csv", default="", help="Optional path to save similarity matrix as CSV")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to show per prompt")
    parser.add_argument("--save-prefix", default="", help="Prefix path to save plots (heatmaps and PCA scatter)")
    parser.add_argument("--mmd-subsample", type=int, default=2048, help="Subsample size per group for MMD/Energy distance")
    parser.add_argument("--shape-metrics", nargs="*", default=["cov_shape", "pair_dist", "pair_angle"], help="Which shape metrics to compute: cov_shape, pair_dist, pair_angle")
    parser.add_argument("--pair-subsample-pairs", type=int, default=50000, help="Number of random within-group pairs for pairwise distance/angle distributions")
    parser.add_argument("--mds-from", choices=["mmd", "energy", "gaussian_w2", "pair_dist", "pair_angle", "cov_shape"], default="mmd", help="Which distance matrix to embed via MDS")
    parser.add_argument("--qq-from", choices=["pair_dist", "pair_angle"], default="pair_dist", help="Which distribution to use for QQ plots")
    parser.add_argument("--qq-top-pairs", type=int, default=6, help="How many closest pairs to show in the QQ grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_prompt_distributions(
        args.pt_path,
        metric="centroid_cosine",
        save_csv_path=args.save_csv or None,
        top_k=args.top_k,
        save_prefix=args.save_prefix or None,
        mmd_subsample=args.mmd_subsample,
        shape_metrics=args.shape_metrics,
        pair_subsample_pairs=args.pair_subsample_pairs,
        mds_from=args.mds_from,
        qq_from=args.qq_from,
        qq_top_pairs=args.qq_top_pairs,
    )


if __name__ == "__main__":
    main() 