#!/usr/bin/env python3
"""Fit MANO to input keypoints (16 or 21 joints) and export mesh + joints JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import numpy as np
from smplx import MANO

# MANO 21-joint layout used by this script:
# wrist + (thumb/index/middle/ring/pinky) * (mcp,pip,dip,tip)
MANO16_FROM_21 = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
MANO_FINGERTIP_VERTS = [744, 320, 443, 555, 672]  # thumb, index, middle, ring, pinky


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a MANO hand model to input keypoints and export an OBJ mesh and JSON joints. "
            "Input joints must be shape (16, 3) or (21, 3)."
        )
    )
    parser.add_argument("--input-json", required=True, help="Path to JSON containing joints.")
    parser.add_argument("--mano-model-path", default="./mano/models", help="Path to MANO models directory.")
    parser.add_argument("--output-obj", default="mano_fit.obj", help="Output OBJ mesh path.")
    parser.add_argument("--output-json", default="mano_fit_joints.json", help="Output keypoints JSON path.")
    parser.add_argument(
        "--output-depth-png",
        default=None,
        help="Optional output depth image (.png). When set, a depth map is rasterized from the fitted mesh.",
    )
    parser.add_argument("--depth-size", type=int, default=512, help="Depth image size in pixels (square).")
    parser.add_argument(
        "--output-joint-format",
        choices=["same", "16", "21", "both"],
        default="both",
        help="Joint format in output JSON. 'same' follows input joint count.",
    )
    parser.add_argument("--hand-side", choices=["left", "right"], default="right")
    parser.add_argument("--iterations", type=int, default=600, help="Optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.03, help="Optimization learning rate.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for optimization (default: auto).",
    )
    parser.add_argument(
        "--no-tip-augmentation",
        action="store_true",
        help=(
            "Disable 16->21 fingertip augmentation. Useful for strict MANO-16 experiments. "
            "When enabled, output formats that require 21 joints will fail if the model only returns 16."
        ),
    )
    return parser.parse_args()


def read_joints(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "joints" not in payload:
            raise ValueError("Input JSON must contain 'joints' key when using object format.")
        joints = payload["joints"]
    elif isinstance(payload, list):
        joints = payload
    else:
        raise ValueError("Input JSON must be either a list or an object with a 'joints' list.")

    tensor = torch.tensor(joints, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"Expected input shape (N, 3); got {tuple(tensor.shape)}")
    if tensor.shape[0] not in (16, 21):
        raise ValueError(f"Expected 16 or 21 joints; got {tensor.shape[0]}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Input joints contain NaN or Inf values.")
    return tensor


def write_obj(path: Path, vertices: torch.Tensor, faces: Iterable[Iterable[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices.tolist():
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_depth_png(path: Path, vertices: torch.Tensor, faces: Iterable[Iterable[int]], size: int = 512) -> None:
    if size <= 0:
        raise ValueError(f"Depth image size must be positive, got {size}.")

    verts = vertices.detach().cpu().numpy().astype(np.float32)
    tris = np.asarray(list(faces), dtype=np.int32)

    xy = verts[:, :2]
    z = verts[:, 2]

    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-8)
    scale = (size - 1) / float(np.max(span))
    xy_pix = (xy - xy_min) * scale
    pad_x = (size - 1 - (xy_max[0] - xy_min[0]) * scale) * 0.5
    pad_y = (size - 1 - (xy_max[1] - xy_min[1]) * scale) * 0.5
    xy_pix[:, 0] += pad_x
    xy_pix[:, 1] += pad_y
    xy_pix[:, 1] = (size - 1) - xy_pix[:, 1]

    zbuf = np.full((size, size), np.inf, dtype=np.float32)

    for i0, i1, i2 in tris:
        p0, p1, p2 = xy_pix[i0], xy_pix[i1], xy_pix[i2]
        z0, z1, z2 = z[i0], z[i1], z[i2]

        min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), size - 1)
        min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), size - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(area) < 1e-8:
            continue

        for yy in range(min_y, max_y + 1):
            for xx in range(min_x, max_x + 1):
                px = xx + 0.5
                py = yy + 0.5

                w0 = ((p1[0] - px) * (p2[1] - py) - (p1[1] - py) * (p2[0] - px)) / area
                w1 = ((p2[0] - px) * (p0[1] - py) - (p2[1] - py) * (p0[0] - px)) / area
                w2 = 1.0 - w0 - w1

                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue

                depth = w0 * z0 + w1 * z1 + w2 * z2
                if depth < zbuf[yy, xx]:
                    zbuf[yy, xx] = depth

    mask = np.isfinite(zbuf)
    depth_u8 = np.zeros((size, size), dtype=np.uint8)
    if np.any(mask):
        valid = zbuf[mask]
        z_min, z_max = float(valid.min()), float(valid.max())
        denom = max(z_max - z_min, 1e-8)
        norm = (valid - z_min) / denom
        depth_u8[mask] = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)

    with path.open("wb") as handle:
        handle.write(f"P5\n{size} {size}\n255\n".encode("ascii"))
        handle.write(depth_u8.tobytes())


def joints16_to_21(joints16: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    if joints16.shape[0] != 16:
        raise ValueError(f"Expected 16 joints, got {joints16.shape[0]}")

    max_tip = max(MANO_FINGERTIP_VERTS)
    if vertices.shape[0] <= max_tip:
        raise ValueError(
            f"Cannot build 21 joints from 16: vertices has {vertices.shape[0]} rows, "
            f"but fingertip vertex id {max_tip} is required."
        )

    joints21 = torch.zeros((21, 3), dtype=joints16.dtype, device=joints16.device)
    joints21[MANO16_FROM_21] = joints16

    # tip locations are sampled from mesh vertices.
    for out_idx, tip_vert_idx in zip([4, 8, 12, 16, 20], MANO_FINGERTIP_VERTS):
        joints21[out_idx] = vertices[tip_vert_idx]

    return joints21


def output_to_joints21(output_joints: torch.Tensor, output_vertices: torch.Tensor) -> torch.Tensor:
    joint_count = int(output_joints.shape[0])

    if joint_count >= 21:
        return output_joints[:21]
    if joint_count == 16:
        return joints16_to_21(output_joints, output_vertices)

    raise ValueError(
        f"Unsupported MANO output joint count {joint_count}. Expected 16 or at least 21."
    )


def output_to_joints16(output_joints: torch.Tensor) -> torch.Tensor:
    joint_count = int(output_joints.shape[0])
    if joint_count == 16:
        return output_joints
    if joint_count >= 21:
        return output_joints[MANO16_FROM_21]
    raise ValueError(f"Unsupported MANO output joint count {joint_count}. Expected 16 or at least 21.")


def fit_mano_to_joints(
    model: MANO,
    target_joints: torch.Tensor,
    iterations: int,
    lr: float,
    no_tip_augmentation: bool = False,
) -> dict[str, torch.Tensor | int | None]:
    device = next(model.parameters()).device
    target_joints = target_joints.to(device=device)

    # Learnable MANO parameters.
    global_orient = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    hand_pose = torch.nn.Parameter(torch.zeros(1, 45, device=device))
    betas = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    transl = torch.nn.Parameter(torch.zeros(1, 3, device=device))

    optimizer = torch.optim.Adam([global_orient, hand_pose, betas, transl], lr=lr)

    target_centered = target_joints - target_joints[0:1]

    for _ in range(iterations):
        optimizer.zero_grad()

        output = model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            transl=transl,
            return_verts=True,
        )

        if target_joints.shape[0] == 16 and no_tip_augmentation:
            predicted_selected = output_to_joints16(output.joints[0])
        elif target_joints.shape[0] == 16:
            predicted_selected = output_to_joints21(output.joints[0], output.vertices[0])[MANO16_FROM_21]
        else:
            predicted_selected = output_to_joints21(output.joints[0], output.vertices[0])
        predicted_centered = predicted_selected - predicted_selected[0:1]

        if predicted_centered.shape != target_centered.shape:
            raise RuntimeError(
                f"Shape mismatch in loss: predicted {tuple(predicted_centered.shape)} vs "
                f"target {tuple(target_centered.shape)}"
            )

        data_loss = ((predicted_centered - target_centered) ** 2).mean()
        pose_reg = (hand_pose ** 2).mean() * 1e-4
        shape_reg = (betas ** 2).mean() * 1e-3

        loss = data_loss + pose_reg + shape_reg
        loss.backward()
        optimizer.step()

    final_output = model(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        transl=transl,
        return_verts=True,
    )

    joints21: torch.Tensor | None
    if int(final_output.joints.shape[1]) >= 21:
        joints21 = output_to_joints21(final_output.joints[0], final_output.vertices[0]).detach().cpu()
    elif no_tip_augmentation:
        joints21 = None
    else:
        joints21 = output_to_joints21(final_output.joints[0], final_output.vertices[0]).detach().cpu()

    return {
        "global_orient": global_orient.detach(),
        "hand_pose": hand_pose.detach(),
        "betas": betas.detach(),
        "transl": transl.detach(),
        "model_output_joint_count_raw": int(final_output.joints.shape[1]),
        "joints16": output_to_joints16(final_output.joints[0]).detach().cpu(),
        "joints21": joints21,
        "vertices": final_output.vertices[0].detach().cpu(),
    }


def build_output_joints(
    joints16: torch.Tensor,
    joints21: torch.Tensor | None,
    requested_format: str,
    input_joint_count: int,
) -> dict[str, list[list[float]] | int | dict[str, int]]:

    def to_list(tensor: torch.Tensor) -> list[list[float]]:
        return tensor.tolist()

    if requested_format == "same":
        requested_format = str(input_joint_count)

    if requested_format == "21":
        if joints21 is None:
            raise ValueError(
                "21-joint output requested but unavailable because --no-tip-augmentation was set "
                "and the loaded MANO returned only 16 joints."
            )
        return {"joints": to_list(joints21), "joint_count": 21}
    if requested_format == "16":
        return {"joints": to_list(joints16), "joint_count": 16}
    if requested_format == "both":
        if joints21 is None:
            raise ValueError(
                "'both' output requested but 21 joints are unavailable because --no-tip-augmentation "
                "was set and the loaded MANO returned only 16 joints."
            )
        return {
            "joints21": to_list(joints21),
            "joints16": to_list(joints16),
            "joint_count": {"joints21": 21, "joints16": 16},
        }

    raise ValueError(f"Unexpected output format: {requested_format}")


def resolve_output_joint_format(requested_format: str, input_joint_count: int) -> str:
    if requested_format == "same":
        return str(input_joint_count)
    return requested_format


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    output_obj_path = Path(args.output_obj)
    output_json_path = Path(args.output_json)

    target_joints = read_joints(input_path)
    device = resolve_device(args.device)

    model = MANO(
        model_path=args.mano_model_path,
        is_rhand=(args.hand_side == "right"),
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to(device)

    result = fit_mano_to_joints(
        model=model,
        target_joints=target_joints,
        iterations=args.iterations,
        lr=args.lr,
        no_tip_augmentation=args.no_tip_augmentation,
    )

    write_obj(output_obj_path, result["vertices"], model.faces)
    depth_png_path = Path(args.output_depth_png) if args.output_depth_png else None
    if depth_png_path is not None:
        write_depth_png(depth_png_path, result["vertices"], model.faces, size=args.depth_size)
    input_joint_count = int(target_joints.shape[0])
    resolved_output_format = resolve_output_joint_format(args.output_joint_format, input_joint_count)
    output_payload = build_output_joints(
        joints16=result["joints16"],
        joints21=result["joints21"],
        requested_format=args.output_joint_format,
        input_joint_count=input_joint_count,
    )

    output_payload["mano_params"] = {
        "global_orient": result["global_orient"][0].cpu().tolist(),
        "hand_pose": result["hand_pose"][0].cpu().tolist(),
        "betas": result["betas"][0].cpu().tolist(),
        "transl": result["transl"][0].cpu().tolist(),
    }
    output_payload["metadata"] = {
        "input_joint_count": input_joint_count,
        "requested_output_joint_format": args.output_joint_format,
        "resolved_output_joint_format": resolved_output_format,
        "model_output_joint_count_raw": result["model_output_joint_count_raw"],
        "tip_augmentation_enabled": not args.no_tip_augmentation,
        "exported_obj_path": str(output_obj_path),
        "exported_depth_png_path": str(depth_png_path) if depth_png_path is not None else None,
    }

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print(f"Using device: {device}")
    print(
        "Joint summary: "
        f"input={input_joint_count}, "
        f"model_raw={result['model_output_joint_count_raw']}, "
        f"json_output={resolved_output_format}"
    )
    print(f"Saved fitted mesh OBJ to {output_obj_path}")
    if depth_png_path is not None:
        print(f"Saved fitted depth image to {depth_png_path}")
    print(f"Saved fitted keypoints JSON to {output_json_path}")


if __name__ == "__main__":
    main()
