# Crash Anticipation Baseline

Baseline training framework for crash anticipation on the Dashcam Accident Dataset (DAD) and Car Crash Dataset (CCD). The goal is to detect hazardous situations a few seconds before impact using pretrained video transformers such as VideoMAE.

## Features

- Video clip sampling utilities for DAD manifests and CCD image sequences with configurable lead-time sampling.
- Preprocessing pipeline for RGB frames (8–12 fps, 224p) with lightweight augmentations.
- VideoMAE-S backbone with a binary anticipation head and configurable optimizer/scheduler.
- Training script with AMP support, gradient accumulation, TensorBoard logging, and checkpointing.
- Evaluation metrics including AP, F1, and lead-time recall.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

### Dashcam Accident Dataset (DAD)

1. Download DAD and organize individual videos or frame folders locally.
2. Generate CSV manifests for each split (`train`, `val`) with at least the following columns:
   - `video_path`: absolute or workspace-relative path to the video file (or directory of frames)
   - `label`: `1` for crash clips, `0` otherwise
   - `event_frame`: index of the crash frame (0-based) for positive samples
   - `fps` *(optional)*: override FPS if different from the dataset default
   - `total_frames` *(optional)*: cached frame count for faster sampling
3. Place manifests under `data/manifests/` (default names in `configs/baseline.yaml`).

### Car Crash Dataset (CCD)

1. Download the CCD release (frames in `CrashBest/` and annotation table `Crash_Table.csv`).
2. Ensure dependencies are installed (pandas, scikit-learn) via `pip install -r requirements.txt`.
3. Generate train/validation manifests:
   ```bash
   python data/manifests/create_ccd_manifests.py \
     --table data/manifests/car-crash-dataset-ccd/Crash_Table.csv \
     --output data/manifests
   ```
   This produces `ccd_train.csv` and `ccd_val.csv` with per-sample metadata and lead-time labels.
4. Keep the extracted frames under `data/manifests/car-crash-dataset-ccd/CrashBest` (default path referenced by the CCD config).

## Training

```bash
python train.py --config configs/baseline.yaml
```

Override configuration values at the command line:

```bash
python train.py --config configs/baseline.yaml --override train.batch_size=8 train.max_epochs=30
```

TensorBoard logs and checkpoints are stored under the configured `output_dir` (default: `outputs/baseline`).

To train on CCD image sequences use the dedicated configuration:

```bash
python train.py --config configs/ccd_baseline.yaml
```

## Project Structure

```
crash_anticipation/
  config.py              # YAML-driven experiment configuration
  data/                  # Dataset loading and transforms
  models/                # VideoMAE baseline model
  training/              # Training loop, metrics, checkpoint helpers
configs/
  baseline.yaml          # Baseline experiment configuration
  ccd_baseline.yaml      # CCD-specific configuration (image sequences)
train.py                 # CLI entry point
```

## Next Steps

- Incorporate additional backbones (TimeSformer, SlowFast) and multi-task heads.
- Add temporal hazard loss functions and curriculum sampling strategies.
- Extend evaluation with time-to-accident AP metrics across longer lead times.

