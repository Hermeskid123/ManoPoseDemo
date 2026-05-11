#!/usr/bin/env python3
"""Fit MANO to input keypoints (16 or 21 joints) and export mesh + joints JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
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
    print(f"Saved fitted keypoints JSON to {output_json_path}")


if __name__ == "__main__":
    main()
