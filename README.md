# S-MFF (3DPW) — `S_MFF_reablation_3dpw` core implementation

This repository contains the **S-MFF core model code from `S_MFF_reablation_3dpw`**.
It keeps the original algorithm and replaces the original server-specific absolute paths with CLI options. No V14 legacy code, RICH experiments, evaluation plotting code, datasets, or model checkpoints are bundled.

## Core source

`src/rebuild_3dpw_reablation_from_manifest.py`

Original experimental-server location:

```text
/root/data1/sam_pose/S_MFF_reablation_3dpw/code/rebuild_3dpw_reablation_from_manifest.py
```

Original SHA256: `b6fd055e074a078b71fc133551714efae3ade3e029f2996e6a0af5eab3e01c94`

The code performs:

1. Load WHAM raw tracked 3DPW features and 3DPW images.
2. Read SAM3 per-frame JSON metadata (not pixel-level mask images).
3. Apply foreground-background attenuation using `mask_bbox_xyxy`.
4. Extract raw and foreground-enhanced WHAM visual features.
5. Compute reliability-adaptive fusion weights and fuse the two feature streams.
6. Optionally apply neighbor context bonus, motion-consistency weight adjustment and temporal feature infilling, according to the manifest.
7. Save modified parsed `.pth` and per-track processing reports.

**Implementation detail:** This 3DPW implementation uses the **SAM3 mask bounding rectangle** from JSON metadata for foreground attenuation. It does not use per-pixel SAM masks in the fusion step.

## Official upstream code and pretrained weights

The pretrained models are **not stored in this GitHub repository**. Download/install them from their respective official release channels and follow the upstream licenses:

- **WHAM** (official code, model checkpoints and setup): https://github.com/yohanshin/WHAM
- **SAM 3** (official segmentation model and checkpoints): https://github.com/facebookresearch/sam3
- **3DPW dataset**: obtain images through the dataset's authorized distribution.

For running the S-MFF fusion stage, WHAM's `FeatureExtractor` and its model weights must be installed under your local WHAM checkout. The SAM3 phase produces the input JSON mask metadata separately; the fusion script directly consumes those JSON files and does **not** load the SAM3 checkpoint.

## Setup

```bash
# In a compatible WHAM environment with PyTorch installed:
pip install -r requirements.txt

# Clone/install WHAM separately; download official WHAM weights as instructed upstream.
# Obtain 3DPW images, WHAM parsed features, and SAM3 mask JSON metadata.
```

## Run

```bash
python src/rebuild_3dpw_reablation_from_manifest.py \
  --wham_root /path/to/WHAM \
  --source_pth /path/to/3dpw_test_vit.raw_backup.pth \
  --image_root /path/to/3DPW/imageFiles \
  --mask_meta_root /path/to/sam3_mask_metadata \
  --manifest configs/example_manifest.csv \
  --exp_ids example_adaptive_temporal_lam018 \
  --output_root ./outputs \
  --batch_size 128
```

Generated artifacts:

```text
outputs/
├── pths/<exp_id>.pth
└── records/<exp_id>_report.csv
```

## Input format

The original script reads a WHAM-parsed `.pth` using `joblib.load()`. Required keys include `vid`, `frame_id`, `bbox`, `features`, and `flipped_features`. Images must be available under `<image_root>/<sequence>/*.jpg`, indexed in lexicographic filename order by the stored integer `frame_id`.

SAM3 metadata is searched at:

```text
<mask_meta_root>/<track_id>/json/<image_stem>.json
<mask_meta_root>/<track_id>/json/image_<frame_id:05d>.json
```

The metadata uses `mask_bbox_xyxy` and `mask_area_full` (or `mask_area`). Missing/unusable metadata triggers the original no-mask fallback. Weights and datasets stay out of the repository through `.gitignore`.

## Source provenance and licensing

The S-MFF implementation derives from `S_MFF_reablation_3dpw`. WHAM and SAM3 are third-party software and are **not** relicensed or redistributed in this repository; follow the respective upstream terms. 
