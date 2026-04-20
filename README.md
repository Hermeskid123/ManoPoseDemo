# MANO Pose Fitting Demo

This project now focuses on fitting a MANO hand model from **either 16 or 21 3D keypoints** and exporting:

- `.obj` mesh
- `.json` keypoints (16, 21, same as input, or both)

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
  --output-joint-format both
```

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

## Notes

- The fitter optimizes MANO pose/shape parameters via gradient descent.
- For 16-joint input, fingertip joints are omitted and recovered through fitting.
- Output JSON also includes fitted MANO parameters (`global_orient`, `hand_pose`, `betas`, `transl`).
