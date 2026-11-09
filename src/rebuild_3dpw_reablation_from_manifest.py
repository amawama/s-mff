import os
import re
import sys
import json
import copy
import argparse
from pathlib import Path

import cv2
import joblib
import torch
import numpy as np
import pandas as pd

# Paths are configurable for public releases. The S-MFF algorithm below is unchanged.
ROOT = Path(__file__).resolve().parents[1]
WHAM_ROOT = Path(os.environ.get("WHAM_ROOT", str(ROOT / "third_party/WHAM")))
SRC = Path(os.environ.get("SMFF_SOURCE_PTH", str(ROOT / "data/3dpw_test_vit.raw_backup.pth")))
IMG_ROOT = Path(os.environ.get("SMFF_3DPW_IMAGE_ROOT", str(ROOT / "data/3DPW/imageFiles")))
MASK_META_ROOT = Path(os.environ.get("SMFF_MASK_META_ROOT", str(ROOT / "data/mask_metadata")))
OUT_PTH_DIR = ROOT / "outputs/pths"
REPORT_DIR = ROOT / "outputs/records"
DEFAULT_MANIFEST = ROOT / "configs/example_manifest.csv"

# WHAM / progress imports are loaded after CLI arguments are parsed.
FeatureExtractor = None
process_image = None
Bar = None


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def base_seq(vid):
    return re.sub(r"_\d+$", "", str(vid))


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def tensor_like(arr, ref):
    if hasattr(ref, "detach"):
        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return np.asarray(arr, dtype=np.asarray(ref).dtype)


def wham_bbox_to_xyxy(bb):
    cx, cy, s = float(bb[0]), float(bb[1]), max(float(bb[2]), 1.0)
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, aa + ba - inter)


def center_dist_norm(a, b):
    acx = (a[0] + a[2]) / 2
    acy = (a[1] + a[3]) / 2
    bcx = (b[0] + b[2]) / 2
    bcy = (b[1] + b[3]) / 2
    scale = max(1.0, a[2] - a[0])
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def build_frame_box_index(data):
    idx = {}
    vids = [str(v) for v in data["vid"]]
    for ii, vid in enumerate(vids):
        seq = base_seq(vid)
        fids = to_np(data["frame_id"][ii]).astype(int)
        bbs = to_np(data["bbox"][ii])
        for fid, bb in zip(fids, bbs):
            key = (seq, int(fid))
            idx.setdefault(key, []).append((vid, wham_bbox_to_xyxy(bb)))
    return idx


def get_neighbor_context(frame_box_index, vid, seq, fid, bb):
    cur = wham_bbox_to_xyxy(bb)
    items = frame_box_index.get((seq, int(fid)), [])

    max_iou = 0.0
    nearby_count = 0
    overlap_count = 0

    for other_vid, other_box in items:
        if other_vid == vid:
            continue
        iou = box_iou(cur, other_box)
        dist = center_dist_norm(cur, other_box)
        max_iou = max(max_iou, iou)
        if iou > 0.03:
            overlap_count += 1
        if dist < 1.5:
            nearby_count += 1

    return {
        "neighbor_max_iou": float(max_iou),
        "neighbor_overlap_count": int(overlap_count),
        "nearby_count": int(nearby_count),
    }


def load_mask_box(vid, fid, image_name):
    stem = Path(str(image_name)).stem
    p1 = MASK_META_ROOT / str(vid) / "json" / f"{stem}.json"
    p2 = MASK_META_ROOT / str(vid) / "json" / f"image_{int(fid):05d}.json"

    for p in [p1, p2]:
        if not p.exists():
            continue
        try:
            with open(p, "r") as f:
                meta = json.load(f)
            box = meta.get("mask_bbox_xyxy", None)
            area = meta.get("mask_area_full", meta.get("mask_area", 0))
            if box is not None and area and int(area) > 64:
                return [float(x) for x in box], int(area)
        except Exception:
            pass

    return None, 0


def v8_masked_weight(mask_box, mask_area, wham_bbox, base_w, min_w, max_w):
    if mask_box is None or mask_area <= 64:
        return 0.0, 999.0, 0.0, False

    x1, y1, x2, y2 = [float(v) for v in mask_box]
    mw = max(1.0, x2 - x1)
    mh = max(1.0, y2 - y1)

    cx_m = (x1 + x2) / 2.0
    cy_m = (y1 + y2) / 2.0

    cx, cy, s = float(wham_bbox[0]), float(wham_bbox[1]), float(wham_bbox[2])
    s = max(s, 1.0)

    center_dist = ((cx_m - cx) ** 2 + (cy_m - cy) ** 2) ** 0.5 / s
    area_ratio = (mw * mh) / max(s * s, 1.0)

    weight = float(base_w)
    reliable = True

    if center_dist < 0.20 and 0.15 <= area_ratio <= 0.85:
        weight = 0.45

    if center_dist > 0.45:
        weight = 0.10
        reliable = False

    if area_ratio < 0.05 or area_ratio > 1.25:
        weight = 0.10
        reliable = False

    if 0.85 < area_ratio <= 1.25:
        weight = min(weight, 0.20)

    weight = float(max(min_w, min(max_w, weight)))
    return weight, float(center_dist), float(area_ratio), reliable


def residual_context_bonus(neighbor_ctx, reliable, area_ratio, level):
    if level == "off":
        return 0.0, "context_off"

    if not reliable:
        return 0.0, "no_bonus_bad_mask"

    if area_ratio > 1.05:
        return 0.0, "no_bonus_large_mask"

    max_iou = neighbor_ctx["neighbor_max_iou"]
    overlap_count = neighbor_ctx["neighbor_overlap_count"]
    nearby_count = neighbor_ctx["nearby_count"]

    if level == "strong":
        mild_bonus, strong_bonus = 0.10, 0.20
    else:
        mild_bonus, strong_bonus = 0.05, 0.10

    bonus = 0.0
    reason = "no_context"

    if max_iou > 0.08 or overlap_count >= 1 or nearby_count >= 2:
        bonus = mild_bonus
        reason = "mild_context_bonus"

    if max_iou > 0.15 or overlap_count >= 2 or nearby_count >= 3:
        bonus = strong_bonus
        reason = "strong_context_bonus"

    return float(bonus), reason


def apply_bbox_foreground_weight(img_rgb, mask_box, alpha):
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in mask_box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return img_rgb

    out = (img_rgb.astype(np.float32) * float(alpha)).astype(np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = img_rgb[y1:y2 + 1, x1:x2 + 1]
    return out


def motion_consistency_adjust_weights(raw_feat, masked_feat, weights, gamma, min_w, max_w):
    if not torch.is_tensor(raw_feat):
        raw_feat = torch.as_tensor(raw_feat).float()
    if not torch.is_tensor(masked_feat):
        masked_feat = torch.as_tensor(masked_feat).float()
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights).float()

    T = raw_feat.shape[0]
    if T < 3 or float(gamma) == 0.0:
        return weights.float().cpu(), 0, 0, 0.0

    raw_d = torch.norm(raw_feat[1:] - raw_feat[:-1], dim=1)
    mask_d = torch.norm(masked_feat[1:] - masked_feat[:-1], dim=1)

    pair_score = (raw_d - mask_d) / (raw_d + mask_d + 1e-6)

    frame_score = torch.zeros(T, dtype=torch.float32)
    count = torch.zeros(T, dtype=torch.float32)
    frame_score[1:] += pair_score.cpu()
    frame_score[:-1] += pair_score.cpu()
    count[1:] += 1
    count[:-1] += 1
    frame_score = frame_score / torch.clamp(count, min=1.0)

    old_w = weights.float().cpu()
    valid = old_w > 0.0

    new_w = old_w.clone()
    new_w[valid] = new_w[valid] + float(gamma) * frame_score[valid]
    new_w[valid] = torch.clamp(new_w[valid], float(min_w), float(max_w))
    new_w[~valid] = 0.0

    diff = new_w - old_w
    inc = int((diff > 1e-5).sum().item())
    dec = int((diff < -1e-5).sum().item())
    mean_abs = float(diff.abs().mean().item())

    return new_w, inc, dec, mean_abs


def temporal_feature_infilling(feat, flip_feat, weights, alpha, max_gap):
    if not torch.is_tensor(feat):
        feat = torch.as_tensor(feat).float()
    if not torch.is_tensor(flip_feat):
        flip_feat = torch.as_tensor(flip_feat).float()
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights).float()

    if float(alpha) <= 0.0 or int(max_gap) <= 0:
        return feat, flip_feat, 0, 0

    T = feat.shape[0]
    if T < 5:
        return feat, flip_feat, 0, 0

    f = feat.float()
    ff = flip_feat.float()

    diffs = torch.norm(f[1:] - f[:-1], dim=1)
    med = torch.median(diffs)
    mad = torch.median(torch.abs(diffs - med)) + 1e-6
    jump_thr = med + 3.0 * mad

    unreliable = weights <= 0.12
    jumpy = torch.zeros(T, dtype=torch.bool)
    jumpy[1:] |= diffs > jump_thr
    jumpy[:-1] |= diffs > jump_thr

    candidates = unreliable | jumpy
    reliable = (~candidates) & (weights >= 0.25)

    new_f = f.clone()
    new_ff = ff.clone()

    repaired = 0
    candidate_count = int(candidates.sum().item())

    for t in range(T):
        if not bool(candidates[t]):
            continue

        left = None
        right = None

        for k in range(1, int(max_gap) + 1):
            if t - k >= 0 and bool(reliable[t - k]):
                left = t - k
                break

        for k in range(1, int(max_gap) + 1):
            if t + k < T and bool(reliable[t + k]):
                right = t + k
                break

        if left is None and right is None:
            continue

        if left is not None and right is not None:
            ref = 0.5 * (f[left] + f[right])
            ref_flip = 0.5 * (ff[left] + ff[right])
        elif left is not None:
            ref = f[left]
            ref_flip = ff[left]
        else:
            ref = f[right]
            ref_flip = ff[right]

        new_f[t] = (1.0 - float(alpha)) * f[t] + float(alpha) * ref
        new_ff[t] = (1.0 - float(alpha)) * ff[t] + float(alpha) * ref_flip
        repaired += 1

    return new_f, new_ff, repaired, candidate_count


def row_cfg(row):
    return {
        "exp_id": str(row["exp_id"]),
        "table_group": str(row.get("table_group", "")),
        "alpha": float(row["alpha"]),
        "weight_mode": str(row["weight_mode"]),
        "fixed_masked_w": float(row["fixed_masked_w"]),
        "base_masked_w": float(row["base_masked_w"]),
        "min_masked_w": float(row["min_masked_w"]),
        "max_masked_w": float(row["max_masked_w"]),
        "context_bonus": str2bool(row["context_bonus"]),
        "context_level": str(row.get("context_level", "default")),
        "motion_adjust": str2bool(row["motion_adjust"]),
        "motion_gamma": float(row["motion_gamma"]),
        "temporal_infill": str2bool(row["temporal_infill"]),
        "temporal_lambda": float(row["temporal_lambda"]),
        "temporal_max_gap": int(row["temporal_max_gap"]),
        "notes": str(row.get("notes", "")),
    }


def compute_base_weight(cfg, mask_box, mask_area, bb):
    mode = cfg["weight_mode"]

    if mode == "raw_only":
        return 0.0, 0.0, 999.0, 0.0, False

    if mask_box is None or mask_area <= 64:
        return 0.0, 0.0, 999.0, 0.0, False

    if mode == "fixed":
        w = float(cfg["fixed_masked_w"])
        x1, y1, x2, y2 = [float(v) for v in mask_box]
        mw = max(1.0, x2 - x1)
        mh = max(1.0, y2 - y1)
        cx_m = (x1 + x2) / 2.0
        cy_m = (y1 + y2) / 2.0
        cx, cy, s = float(bb[0]), float(bb[1]), max(float(bb[2]), 1.0)
        center_dist = ((cx_m - cx) ** 2 + (cy_m - cy) ** 2) ** 0.5 / s
        area_ratio = (mw * mh) / max(s * s, 1.0)
        return w, w, float(center_dist), float(area_ratio), True

    if mode == "adaptive_v8":
        w, center_dist, area_ratio, reliable = v8_masked_weight(
            mask_box,
            mask_area,
            bb,
            cfg["base_masked_w"],
            cfg["min_masked_w"],
            cfg["max_masked_w"],
        )
        return w, w, center_dist, area_ratio, reliable

    raise ValueError(f"Unknown weight_mode: {mode}")


def save_atomic_joblib(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    joblib.dump(obj, tmp)
    tmp.replace(path)


def save_atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def process_raw_copy(cfg, data, force=False):
    exp_id = cfg["exp_id"]
    out_pth = OUT_PTH_DIR / f"{exp_id}.pth"
    out_report = REPORT_DIR / f"{exp_id}_report.csv"

    if out_pth.exists() and out_report.exists() and not force:
        print(f"[SKIP] {exp_id}: outputs exist", flush=True)
        return

    print(f"[RAW COPY] {exp_id}", flush=True)
    new_data = copy.deepcopy(data)
    save_atomic_joblib(new_data, out_pth)

    rows = []
    for vid, fids in zip(data["vid"], data["frame_id"]):
        fids = to_np(fids)
        rows.append({
            "exp_id": exp_id,
            "vid": str(vid),
            "seq": base_seq(str(vid)),
            "frames": int(len(fids)),
            "used_mask_frames": 0,
            "missing_mask_frames": 0,
            "mean_base_w": 0.0,
            "mean_final_w": 0.0,
            "min_final_w": 0.0,
            "max_final_w": 0.0,
            "motion_inc_frames": 0,
            "motion_dec_frames": 0,
            "motion_mean_abs_delta": 0.0,
            "temporal_repaired_frames": 0,
            "temporal_candidate_frames": 0,
            **{k: cfg[k] for k in cfg if k not in {"notes"}},
        })
    save_atomic_csv(pd.DataFrame(rows), out_report)

    print("saved:", out_pth, flush=True)
    print("saved:", out_report, flush=True)


def process_feature_exp(cfg, data, frame_box_index, extractor, device, batch_size, force=False):
    exp_id = cfg["exp_id"]
    out_pth = OUT_PTH_DIR / f"{exp_id}.pth"
    out_report = REPORT_DIR / f"{exp_id}_report.csv"

    if out_pth.exists() and out_report.exists() and not force:
        print(f"[SKIP] {exp_id}: outputs exist", flush=True)
        return

    print("\n============================================================", flush=True)
    print("EXP:", exp_id, flush=True)
    print("CFG:", cfg, flush=True)
    print("============================================================", flush=True)

    new_data = copy.deepcopy(data)
    vids = [str(v) for v in data["vid"]]
    rows = []

    with torch.no_grad():
        for n, vid in enumerate(vids, 1):
            seq = base_seq(vid)
            print(f"\n[{n}/{len(vids)}] {vid}", flush=True)

            frame_ids = to_np(data["frame_id"][n - 1]).astype(int)
            bbox = to_np(data["bbox"][n - 1]).copy()

            img_paths_all = sorted((IMG_ROOT / seq).glob("*.jpg"))
            if not img_paths_all:
                raise FileNotFoundError(f"missing images for seq={seq}: {IMG_ROOT / seq}")

            raw_patches = []
            masked_patches = []
            masked_weights = []
            base_weights = []
            context_bonuses = []
            neighbor_iou_list = []
            nearby_count_list = []
            overlap_count_list = []
            center_dists = []
            area_ratios = []
            reason_counts = {}

            used_mask = 0
            missing_mask = 0

            bar = Bar(f"Build patches {exp_id} {vid}", fill="#", max=len(frame_ids))

            for fid, bb in zip(frame_ids, bbox):
                fid_int = int(fid)
                if fid_int < 0 or fid_int >= len(img_paths_all):
                    raise IndexError(f"fid out of range: vid={vid}, fid={fid_int}, n_imgs={len(img_paths_all)}")

                img_path = img_paths_all[fid_int]
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    raise FileNotFoundError(str(img_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                raw_patch, _ = process_image(img_rgb, bb[:2], bb[2] / 200, 256, 256)

                mask_box, area = load_mask_box(vid, fid_int, img_path.name)
                neighbor_ctx = get_neighbor_context(frame_box_index, vid, seq, fid_int, bb)

                if mask_box is not None:
                    used_mask += 1
                    img_masked = apply_bbox_foreground_weight(img_rgb, mask_box, cfg["alpha"])
                    base_w, w_before_context, center_dist, area_ratio, reliable = compute_base_weight(cfg, mask_box, area, bb)

                    if cfg["context_bonus"]:
                        bonus, reason = residual_context_bonus(
                            neighbor_ctx,
                            reliable,
                            area_ratio,
                            cfg["context_level"],
                        )
                    else:
                        bonus, reason = 0.0, "context_off"

                    if cfg["weight_mode"] == "fixed":
                        masked_w = base_w
                    else:
                        masked_w = float(max(cfg["min_masked_w"], min(cfg["max_masked_w"], base_w + bonus)))
                else:
                    missing_mask += 1
                    img_masked = img_rgb
                    base_w, masked_w = 0.0, 0.0
                    center_dist, area_ratio, reliable = 999.0, 0.0, False
                    bonus, reason = 0.0, "missing_mask"

                masked_patch, _ = process_image(img_masked, bb[:2], bb[2] / 200, 256, 256)

                raw_patches.append(torch.from_numpy(raw_patch).unsqueeze(0).float())
                masked_patches.append(torch.from_numpy(masked_patch).unsqueeze(0).float())

                base_weights.append(float(base_w))
                masked_weights.append(float(masked_w))
                context_bonuses.append(float(bonus))
                neighbor_iou_list.append(float(neighbor_ctx["neighbor_max_iou"]))
                nearby_count_list.append(int(neighbor_ctx["nearby_count"]))
                overlap_count_list.append(int(neighbor_ctx["neighbor_overlap_count"]))
                center_dists.append(float(center_dist))
                area_ratios.append(float(area_ratio))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

                bar.next()

            bar.finish()

            raw_patches = torch.cat(raw_patches, dim=0)
            masked_patches = torch.cat(masked_patches, dim=0)
            masked_weights = torch.tensor(masked_weights, dtype=torch.float32)
            base_weights_t = torch.tensor(base_weights, dtype=torch.float32)

            raw_features = []
            masked_features = []
            raw_flipped_features = []
            masked_flipped_features = []

            raw_chunks = torch.split(raw_patches, batch_size)
            masked_chunks = torch.split(masked_patches, batch_size)

            bar2 = Bar(f"Encode features {exp_id} {vid}", fill="#", max=len(raw_chunks))

            for raw_patch, masked_patch in zip(raw_chunks, masked_chunks):
                raw_patch = raw_patch.to(device)
                masked_patch = masked_patch.to(device)

                raw_feat = extractor.model(raw_patch, encode=True)
                masked_feat = extractor.model(masked_patch, encode=True)

                raw_features.append(raw_feat.cpu())
                masked_features.append(masked_feat.cpu())

                raw_flip_feat = extractor.model(torch.flip(raw_patch, (3,)), encode=True)
                masked_flip_feat = extractor.model(torch.flip(masked_patch, (3,)), encode=True)

                raw_flipped_features.append(raw_flip_feat.cpu())
                masked_flipped_features.append(masked_flip_feat.cpu())

                bar2.next()

            bar2.finish()

            raw_features = torch.cat(raw_features, dim=0)
            masked_features = torch.cat(masked_features, dim=0)
            raw_flipped_features = torch.cat(raw_flipped_features, dim=0)
            masked_flipped_features = torch.cat(masked_flipped_features, dim=0)

            motion_inc_frames = 0
            motion_dec_frames = 0
            motion_mean_abs_delta = 0.0

            if cfg["motion_adjust"]:
                masked_weights, motion_inc_frames, motion_dec_frames, motion_mean_abs_delta = motion_consistency_adjust_weights(
                    raw_features,
                    masked_features,
                    masked_weights,
                    gamma=cfg["motion_gamma"],
                    min_w=cfg["min_masked_w"],
                    max_w=cfg["max_masked_w"],
                )

            w_all = masked_weights.view(-1, 1)
            fused_features = (1.0 - w_all) * raw_features + w_all * masked_features
            fused_flipped = (1.0 - w_all) * raw_flipped_features + w_all * masked_flipped_features

            temporal_repaired = 0
            temporal_candidates = 0

            if cfg["temporal_infill"]:
                fused_features, fused_flipped, temporal_repaired, temporal_candidates = temporal_feature_infilling(
                    fused_features,
                    fused_flipped,
                    masked_weights,
                    alpha=cfg["temporal_lambda"],
                    max_gap=cfg["temporal_max_gap"],
                )

            new_data["features"][n - 1] = tensor_like(fused_features.numpy(), data["features"][n - 1])
            new_data["flipped_features"][n - 1] = tensor_like(fused_flipped.numpy(), data["flipped_features"][n - 1])

            rows.append({
                "exp_id": exp_id,
                "vid": vid,
                "seq": seq,
                "frames": int(len(frame_ids)),
                "used_mask_frames": int(used_mask),
                "missing_mask_frames": int(missing_mask),
                "mean_base_w": float(base_weights_t.mean().item()) if len(base_weights_t) else 0.0,
                "mean_context_bonus": float(np.mean(context_bonuses)) if context_bonuses else 0.0,
                "mean_final_w": float(masked_weights.mean().item()) if len(masked_weights) else 0.0,
                "min_final_w": float(masked_weights.min().item()) if len(masked_weights) else 0.0,
                "max_final_w": float(masked_weights.max().item()) if len(masked_weights) else 0.0,
                "mean_center_dist": float(np.mean(center_dists)) if center_dists else 0.0,
                "mean_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
                "mean_neighbor_iou": float(np.mean(neighbor_iou_list)) if neighbor_iou_list else 0.0,
                "mean_nearby_count": float(np.mean(nearby_count_list)) if nearby_count_list else 0.0,
                "mean_overlap_count": float(np.mean(overlap_count_list)) if overlap_count_list else 0.0,
                "motion_inc_frames": int(motion_inc_frames),
                "motion_dec_frames": int(motion_dec_frames),
                "motion_mean_abs_delta": float(motion_mean_abs_delta),
                "temporal_repaired_frames": int(temporal_repaired),
                "temporal_candidate_frames": int(temporal_candidates),
                "reason_counts_json": json.dumps(reason_counts, ensure_ascii=False, sort_keys=True),
                **{k: cfg[k] for k in cfg if k not in {"notes"}},
            })

            print(
                "summary:",
                "used_mask", used_mask,
                "missing", missing_mask,
                "base_w", float(base_weights_t.mean().item()) if len(base_weights_t) else 0.0,
                "final_w", float(masked_weights.mean().item()) if len(masked_weights) else 0.0,
                "motion_inc", motion_inc_frames,
                "motion_dec", motion_dec_frames,
                "temporal_repaired", temporal_repaired,
                flush=True,
            )

    save_atomic_joblib(new_data, out_pth)
    save_atomic_csv(pd.DataFrame(rows), out_report)

    print("saved:", out_pth, flush=True)
    print("saved:", out_report, flush=True)


def main():
    global WHAM_ROOT, SRC, IMG_ROOT, MASK_META_ROOT, OUT_PTH_DIR, REPORT_DIR
    global FeatureExtractor, process_image, Bar
    parser = argparse.ArgumentParser(description="S-MFF 3DPW feature reconstruction / reliability fusion")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exp_ids", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wham_root", type=Path, default=WHAM_ROOT,
                        help="Path to the separately installed official WHAM checkout")
    parser.add_argument("--source_pth", type=Path, default=SRC,
                        help="WHAM parsed 3DPW feature PTH")
    parser.add_argument("--image_root", type=Path, default=IMG_ROOT,
                        help="3DPW imageFiles directory")
    parser.add_argument("--mask_meta_root", type=Path, default=MASK_META_ROOT,
                        help="SAM3 mask JSON metadata root (track/json/image_XXXXX.json)")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    WHAM_ROOT = args.wham_root.expanduser().resolve()
    SRC = args.source_pth.expanduser().resolve()
    IMG_ROOT = args.image_root.expanduser().resolve()
    MASK_META_ROOT = args.mask_meta_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    OUT_PTH_DIR = output_root / "pths"
    REPORT_DIR = output_root / "records"

    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not WHAM_ROOT.is_dir():
        raise FileNotFoundError(f"WHAM source directory not found: {WHAM_ROOT}")

    OUT_PTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WHAM_ROOT)
    if str(WHAM_ROOT) not in sys.path:
        sys.path.insert(0, str(WHAM_ROOT))

    from lib.models.preproc.extractor import FeatureExtractor as _FeatureExtractor
    from lib.models.preproc.backbone.utils import process_image as _process_image
    from progress.bar import Bar as _Bar
    FeatureExtractor, process_image, Bar = _FeatureExtractor, _process_image, _Bar

    print("WHAM_ROOT:", WHAM_ROOT, flush=True)
    print("SRC:", SRC, flush=True)
    print("IMG_ROOT:", IMG_ROOT, flush=True)
    print("MASK_META_ROOT:", MASK_META_ROOT, flush=True)
    print("manifest:", manifest, flush=True)

    df = pd.read_csv(manifest)
    if args.exp_ids:
        wanted = set(args.exp_ids)
        df = df[df["exp_id"].isin(wanted)].copy()
        missing = wanted - set(df["exp_id"])
        if missing:
            raise ValueError(f"exp_ids not found in manifest: {sorted(missing)}")

    if len(df) == 0:
        raise RuntimeError("No experiments selected.")

    print("selected experiments:", list(df["exp_id"]), flush=True)
    print("n_experiments:", len(df), flush=True)

    print("loading raw source...", flush=True)
    data = joblib.load(SRC)
    print("tracks:", len(data["vid"]), flush=True)

    print("building frame neighbor index...", flush=True)
    frame_box_index = build_frame_box_index(data)
    print("frame neighbor index ready:", len(frame_box_index), flush=True)

    need_model = any(str(x) != "raw_only" for x in df["weight_mode"].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    extractor = None
    if need_model:
        extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=args.batch_size)

    for _, row in df.iterrows():
        cfg = row_cfg(row)

        if cfg["weight_mode"] == "raw_only":
            process_raw_copy(cfg, data, force=args.force)
        else:
            process_feature_exp(
                cfg,
                data,
                frame_box_index,
                extractor,
                device,
                batch_size=args.batch_size,
                force=args.force,
            )

    print("ALL DONE.", flush=True)
    print("pth dir:", OUT_PTH_DIR, flush=True)
    print("report dir:", REPORT_DIR, flush=True)


if __name__ == "__main__":
    main()