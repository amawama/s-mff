# Local inputs (not committed)

The model script expects the following externally supplied resources:

- `3dpw_test_vit.raw_backup.pth`: WHAM parsed raw 3DPW features.
- `3DPW/imageFiles/<sequence>/*.jpg`: licensed 3DPW test frames.
- `mask_metadata/<track>/json/image_XXXXX.json`: SAM3-generated metadata, including `mask_bbox_xyxy` and `mask_area_full` (or `mask_area`).

Point the command-line flags at your actual paths; no data files need to be copied into this repo.
