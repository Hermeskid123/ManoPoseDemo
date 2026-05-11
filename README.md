# MANO Pose Fitting Demo

This project now focuses on fitting a MANO hand model from **either 16 or 21 3D keypoints** and exporting:

- `.obj` mesh
- `.json` keypoints (16, 21, same as input, or both)
- optional grayscale depth image (`.png` path, PGM data) rasterized from the fitted mesh
- RGB color-coded depth image (`.png`) using white → red → green → blue mapping

## Requirements

- Python 3.9+
- `torch`
- `smplx`
- MANO model files available under `./mano/models` (or pass a custom path)

## Input JSON format

You can provide either:

1) A raw list of points:

```json
[[x, y, z], [x, y, z], ...]
```

2) An object with a `joints` field:

```json
{
  "joints": [[x, y, z], [x, y, z], ...]
}
```

Point count must be exactly **16** or **21**.

## Usage

```bash
python fit_mano.py \
  --input-json input_joints.json \
  --mano-model-path ./mano/models \
  --output-obj fitted_hand.obj \
  --output-json fitted_hand.json \
  --output-joint-format both \
  --device auto
```

### Strict MANO-16 experiment (no 21 tip generation)

If you want to test with only 16 joints and avoid generating fingertip-based 21 joints:

```bash
python fit_mano.py \
  --input-json input_16_joints.json \
  --output-joint-format 16 \
  --no-tip-augmentation
```

### Depth image export (enabled by default)

A depth map is exported by default to `mano_fit_depth.png`. You can override it with `--output-depth-png`:

```bash
python fit_mano.py \
  --input-json input_joints.json \
  --output-depth-png fitted_depth.png \
  --depth-size 512 \
  --output-depth-rgb-png fitted_depth_rgb.png
```

- `--output-depth-png` sets the depth output filename/path (default: `mano_fit_depth.png`).
- `--depth-size` controls output resolution (square image).
- `--output-depth-rgb-png` sets RGB color depth output path (default: `mano_fit_depth_rgb.png`).
- `--depth-max-limit` optionally clamps far depth values before color coding.

### Output joint formats

- `--output-joint-format same` → output matches input count
- `--output-joint-format 16` → output only 16 joints
- `--output-joint-format 21` → output only 21 joints
- `--output-joint-format both` → output both joint sets

## Quick start wrapper (auto-fills --input-json)

If you don't want to pass `--input-json` every run, use:

```bash
./run_fit_mano.sh
```

This wrapper will:
- auto-fill `--input-json sample_joints21.json` when missing
- create `sample_joints21.json` if it does not exist
- forward all other CLI flags to `fit_mano.py`

Example:

```bash
./run_fit_mano.sh --output-joint-format same --iterations 300
```

## How to tell whether your output is 16 or 21 keypoints

After each run, check the output JSON (`--output-json`, default `mano_fit_joints.json`):

- `metadata.input_joint_count` tells you whether your input file had 16 or 21 joints.
- `metadata.resolved_output_joint_format` tells you what the JSON output was written as (`16`, `21`, or `both`).
- `metadata.tip_augmentation_enabled` tells you whether fingertip-based 16→21 augmentation was enabled.
- `joint_count` reports the exported joint count(s).

Additionally, the CLI prints a `Joint summary:` line with input count, raw MANO model joint count, and JSON output format.

## Notes

- If you see CUDA index/assert errors, run with `--device cpu` to avoid asynchronous CUDA kernel failures while debugging.
- The fitter optimizes MANO pose/shape parameters via gradient descent.
- For 16-joint input, fingertip joints are omitted and recovered through fitting.
- Output JSON also includes fitted MANO parameters (`global_orient`, `hand_pose`, `betas`, `transl`).
