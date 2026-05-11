#!/usr/bin/env python3
"""Convert a hand .obj mesh into a point-cloud graph.

Given an OBJ file (for example random_mano_hand.obj), this script reads mesh vertices,
optionally downsamples them, then builds a k-nearest-neighbor graph over the point cloud.

Outputs:
- <prefix>_points.npy: point cloud array of shape (N, 3)
- <prefix>_edges.npy: graph edge list of shape (E, 2)
- <prefix>_graph.npz: compressed bundle with points + edges + metadata
- (optional) <prefix>_graph.png: 3D visualization
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """Load only vertex ('v ') records from an OBJ file."""
    vertices = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    if not vertices:
        raise ValueError(f"No OBJ vertices found in: {obj_path}")
    return np.asarray(vertices, dtype=np.float32)


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    """Build an undirected k-NN edge list from Nx3 points."""
    n = len(points)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    k = max(1, min(k, n - 1))

    # Pairwise squared distances: shape (N, N)
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)

    # Exclude self by setting diagonal to +inf
    np.fill_diagonal(dist2, np.inf)

    nbrs = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]

    edge_set = set()
    for i in range(n):
        for j in nbrs[i]:
            a, b = sorted((int(i), int(j)))
            edge_set.add((a, b))

    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    return edges


def save_outputs(points: np.ndarray, edges: np.ndarray, output_prefix: Path, k: int) -> Tuple[Path, Path, Path]:
    points_path = output_prefix.with_name(output_prefix.name + "_points.npy")
    edges_path = output_prefix.with_name(output_prefix.name + "_edges.npy")
    graph_path = output_prefix.with_name(output_prefix.name + "_graph.npz")

    np.save(points_path, points)
    np.save(edges_path, edges)
    np.savez_compressed(
        graph_path,
        points=points,
        edges=edges,
        num_points=len(points),
        num_edges=len(edges),
        knn_k=k,
    )
    return points_path, edges_path, graph_path


def save_plot(points: np.ndarray, edges: np.ndarray, output_png: Path, point_size: float = 8.0) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, alpha=0.9)

    # Draw edge lines (lightweight for moderate edge counts)
    for i, j in edges:
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], linewidth=0.5, alpha=0.4)

    ax.set_title("Point Cloud Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Keep aspect ratio roughly equal
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert OBJ mesh to point cloud graph")
    p.add_argument("obj_path", help="Input OBJ mesh (e.g. random_mano_hand.obj)")
    p.add_argument("--output-prefix", default="mano_pointcloud", help="Output filename prefix")
    p.add_argument("--k", type=int, default=8, help="k in k-nearest-neighbor graph")
    p.add_argument("--max-points", type=int, default=0, help="Optional random downsample cap (0 = disabled)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
    p.add_argument("--plot", action="store_true", help="Also save a 3D graph visualization PNG")
    p.add_argument("--plot-path", default="", help="Optional custom output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    obj_path = Path(args.obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"Input OBJ not found: {obj_path}")

    points = load_obj_vertices(obj_path)
    points = downsample_points(points, max_points=args.max_points, seed=args.seed)
    edges = knn_graph(points, k=args.k)

    out_prefix = Path(args.output_prefix)
    points_path, edges_path, graph_path = save_outputs(points, edges, out_prefix, k=args.k)

    print(f"Loaded vertices: {len(points)}")
    print(f"Graph edges: {len(edges)} (k={args.k})")
    print(f"Saved points: {points_path}")
    print(f"Saved edges:  {edges_path}")
    print(f"Saved bundle: {graph_path}")

    if args.plot:
        if args.plot_path:
            plot_path = Path(args.plot_path)
        else:
            plot_path = out_prefix.with_name(out_prefix.name + "_graph.png")
        save_plot(points, edges, plot_path)
        print(f"Saved plot:   {plot_path}")


if __name__ == "__main__":
    main()
