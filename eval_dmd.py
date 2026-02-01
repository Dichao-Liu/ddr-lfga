#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import types
import json
import random
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

# ============================== CONFIG =================================== #
# You specify these three paths at the top.
ROOT_JSON_DIR   = "/path/to/dmd"
ROOT_FRAMES_DIR = "/path/to/dmd_frames"
CKPT_PROPOSED   = "/path/to/model.pth" #Please use the weights trained on AUC-DDD.

BATCH_SIZE = 256
SAMPLES_PER_CLASS = 40
NUM_RUNS = 5  # number of independent random runs for mean/std

FRAME_NAME = "frame_{:06d}.jpg"

# ====================== Fixed 10-class taxonomy ========================== #
IDX_TO_CLASS = {
    0: 'drinking',
    1: 'hair_and_makeup',
    2: 'operating_the_radio',
    3: 'reaching_behind',
    4: 'safe_driving',
    5: 'talking_on_the_phone_left',
    6: 'talking_on_the_phone_right',
    7: 'talking_to_passenger',
    8: 'texting_left',
    9: 'texting_right',
}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

# -------------------- Mapping (dataset leaf -> 10-way taxonomy) -------------------- #
# IMPORTANT: reach_side and standstill_or_waiting are intentionally not mapped.
DS_TO_MODEL = {
    "drinking": "drinking",
    "hair_and_makeup": "hair_and_makeup",
    "operating_the_radio": "operating_the_radio",
    "reaching_behind": "reaching_behind",
    "talking_to_passenger": "talking_to_passenger",
    "texting_left": "texting_left",
    "texting_right": "texting_right",
    "safe_driving": "safe_driving",
    "talking_on_the_phone_left": "talking_on_the_phone_left",
    "talking_on_the_phone_right": "talking_on_the_phone_right",
    # aliases seen in raw labels
    "safe_drive": "safe_driving",
    "phonecall_left": "talking_on_the_phone_left",
    "phonecall_right": "talking_on_the_phone_right",
    "radio": "operating_the_radio",
    "reach_backseat": "reaching_behind",
}

# ============================= OpenLabel utils =========================== #
def _leaf(name: str) -> str:
    return name.split("/")[-1] if name else ""

def _collect_intervals_by_type(actions_top: dict) -> Dict[str, List[Tuple[int, int]]]:
    d: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for a in (actions_top or {}).values():
        t = a.get("type", "") or ""
        for itv in (a.get("frame_intervals", []) or []):
            s = itv.get("frame_start")
            e = itv.get("frame_end")
            if s is None or e is None:
                continue
            try:
                s = int(s); e = int(e)
            except Exception:
                continue
            if e < s:
                s, e = e, s
            d[t].append((s, e))
    return d

def _infer_num_frames(frames_dict: dict, intervals_by_type: Dict[str, List[Tuple[int, int]]]) -> int:
    max_frame = -1
    for k in (frames_dict or {}).keys():
        try:
            max_frame = max(max_frame, int(k))
        except Exception:
            pass
    for ivals in intervals_by_type.values():
        for s, e in ivals:
            max_frame = max(max_frame, s, e)
    return (max_frame + 1) if max_frame >= 0 else 0

def _build_per_frame_sets(intervals_by_type: Dict[str, List[Tuple[int, int]]],
                          num_frames: int,
                          prefix: str) -> List[Set[str]]:
    per_frame: List[Set[str]] = [set() for _ in range(num_frames)]
    for t, ivals in intervals_by_type.items():
        if not t.startswith(prefix):
            continue
        v = _leaf(t)
        for s, e in ivals:
            if num_frames <= 0:
                continue
            s = max(0, min(s, num_frames - 1))
            e = max(0, min(e, num_frames - 1))
            if e < s:
                s, e = e, s
            for i in range(s, e + 1):
                per_frame[i].add(v)
    return per_frame

def _segments_from_per_frame(pf: List[Set[str]]) -> List[Tuple[int, int, Set[str]]]:
    segs: List[Tuple[int, int, Set[str]]] = []
    if not pf:
        return segs
    cur = set(pf[0]); start = 0
    for i in range(1, len(pf)):
        if pf[i] != cur:
            segs.append((start, i - 1, set(cur)))
            start = i; cur = set(pf[i])
    segs.append((start, len(pf) - 1, set(cur)))
    return segs

def load_segments_all_from_json(json_path: str) -> List[Tuple[int, int, Set[str]]]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    ol = data.get("openlabel", {}) or {}
    frames = ol.get("frames", {}) or {}
    actions_top = ol.get("actions", {}) or {}
    ivals = _collect_intervals_by_type(actions_top)
    num_frames = _infer_num_frames(frames, ivals)
    pf = _build_per_frame_sets(ivals, num_frames, prefix="driver_actions/")
    segs = _segments_from_per_frame(pf)
    return segs

# ============================ Frames directory ============================ #
def _derive_frames_parent_dir(json_path: str) -> Path:
    # Map the JSON path (relative to ROOT_JSON_DIR) into the frames tree under ROOT_FRAMES_DIR
    rel = os.path.relpath(os.path.dirname(json_path), ROOT_JSON_DIR)
    return Path(ROOT_FRAMES_DIR) / rel

def _derive_expected_frames_dirname_from_json(json_filename: str) -> Optional[str]:
    # JSON file stem like: gA_1_s1_..._rgb_ann_distraction.json
    stem = Path(json_filename).stem
    suffix = "_rgb_ann_distraction"
    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
        return f"{base}_rgb_mosaic_body_frames_256"
    return None

def locate_frames_dir_for_json(json_path: str) -> Optional[Path]:
    # Minimal and necessary logic to find the frames folder that contains frame_XXXXXX.jpg
    parent = _derive_frames_parent_dir(json_path)
    expected = _derive_expected_frames_dirname_from_json(os.path.basename(json_path))
    if expected:
        p = parent / expected
        if p.is_dir():
            return p
    # fallback: find a dir that matches the segment prefix and contains "rgb_mosaic_body_frames"
    stem = Path(json_path).stem
    prefix = stem.split("_rgb_")[0] if "_rgb_" in stem else stem
    if parent.is_dir():
        for entry in parent.iterdir():
            if entry.is_dir():
                name = entry.name
                if name.startswith(prefix) and "rgb_mosaic_body_frames" in name:
                    return entry
    return None

def segment_frames_exist(frames_dir: Path, start: int, end: int) -> bool:
    for i in range(start, end + 1):
        if not (frames_dir / (FRAME_NAME.format(i))).is_file():
            return False
    return True

# ======================== Scan + collect kept segments ==================== #
def find_all_ann_jsons(root_dir: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("rgb_ann_distraction.json"):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)

def scan_and_collect(root_dir: str):
    json_files = find_all_ann_jsons(root_dir)

    kept_counts = {cls: 0 for cls in IDX_TO_CLASS.values()}
    kept_segments_by_class: Dict[str, List[dict]] = {cls: [] for cls in IDX_TO_CLASS.values()}

    pre_single_mapped = {cls: 0 for cls in IDX_TO_CLASS.values()}
    drop_missing_frames = {cls: 0 for cls in IDX_TO_CLASS.values()}
    drop_frames_dir_missing = {cls: 0 for cls in IDX_TO_CLASS.values()}
    multi_label_involved = {cls: 0 for cls in IDX_TO_CLASS.values()}

    unmapped_single = defaultdict(int)
    unmapped_multi = defaultdict(int)

    total_json = 0
    total_segments_all = 0
    total_single_label = 0
    total_multi_label = 0
    json_without_frames_dir = 0

    for jp in json_files:
        try:
            segs_all = load_segments_all_from_json(jp)
        except Exception as e:
            print(f"[WARN] Skip JSON due to parse error: {jp} ({e})")
            continue

        total_json += 1
        total_segments_all += len(segs_all)

        # diagnostics before frame filtering
        for s, e, labset in segs_all:
            if len(labset) == 1:
                total_single_label += 1
                ds_lab = next(iter(labset))
                model_lab = DS_TO_MODEL.get(ds_lab, ds_lab if ds_lab in CLASS_TO_IDX else None)
                if model_lab is None:
                    unmapped_single[ds_lab] += 1
                else:
                    pre_single_mapped[model_lab] += 1
            else:
                total_multi_label += 1
                for ds_lab in labset:
                    model_lab = DS_TO_MODEL.get(ds_lab, ds_lab if ds_lab in CLASS_TO_IDX else None)
                    if model_lab is None:
                        unmapped_multi[ds_lab] += 1
                    else:
                        multi_label_involved[model_lab] += 1

        frames_dir = locate_frames_dir_for_json(jp)
        if frames_dir is None or not frames_dir.is_dir():
            json_without_frames_dir += 1
            for s, e, labset in segs_all:
                if len(labset) != 1:
                    continue
                ds_lab = next(iter(labset))
                model_lab = DS_TO_MODEL.get(ds_lab, ds_lab if ds_lab in CLASS_TO_IDX else None)
                if model_lab in drop_frames_dir_missing:
                    drop_frames_dir_missing[model_lab] += 1
            continue

        for s, e, labset in segs_all:
            if len(labset) != 1:
                continue
            ds_lab = next(iter(labset))
            model_lab = DS_TO_MODEL.get(ds_lab, ds_lab if ds_lab in CLASS_TO_IDX else None)
            if model_lab not in kept_counts:
                continue
            if segment_frames_exist(frames_dir, s, e):
                kept_counts[model_lab] += 1
                kept_segments_by_class[model_lab].append({
                    "frames_dir": str(frames_dir),
                    "start": s,
                    "end": e,
                    "ds_label": model_lab
                })
            else:
                drop_missing_frames[model_lab] += 1

    # diagnostics
    print(f"[INFO] JSON files scanned: {total_json}")
    print(f"[INFO] Segments (all labels, before filtering): {total_segments_all}")
    print(f"[INFO] Single-label segments: {total_single_label}")
    print(f"[INFO] Multi-label segments: {total_multi_label}")
    print(f"[INFO] JSONs without frames directory: {json_without_frames_dir}")
    print(f"[INFO] Mapping notes: reach_side -> (no map), standstill_or_waiting -> (no map)\n")

    header = "{:<28s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "class", "pre_map", "kept", "miss_frm", "no_dir", "multi_in"
    )
    print("=== Per-class diagnostics (10-way) ===")
    print(header)
    print("-" * len(header))
    for i in range(10):
        name = IDX_TO_CLASS[i]
        print("{:<28s} {:>8d} {:>8d} {:>8d} {:>8d} {:>8d}".format(
            name,
            pre_single_mapped.get(name, 0),
            kept_counts.get(name, 0),
            drop_missing_frames.get(name, 0),
            drop_frames_dir_missing.get(name, 0),
            multi_label_involved.get(name, 0),
        ))

    if unmapped_single:
        print("\n=== Unmapped single-label segments (raw dataset labels) ===")
        items = sorted(unmapped_single.items(), key=lambda x: (-x[1], x[0]))
        for k, v in items[:100]:
            print(f"{k:40s} : {v}")
        if len(items) > 100:
            print(f"... ({len(items) - 100} more)")

    if unmapped_multi:
        print("\n=== Unmapped labels that appear in multi-label segments (raw labels) ===")
        items = sorted(unmapped_multi.items(), key=lambda x: (-x[1], x[0]))
        for k, v in items[:100]:
            print(f"{k:40s} : {v}")
        if len(items) > 100:
            print(f"... ({len(items) - 100} more)")

    print("\n=== Final kept counts (10-way) ===")
    for i in range(10):
        name = IDX_TO_CLASS[i]
        print(f"{i:02d}  {name:25s} : {kept_counts.get(name, 0)}")

    return kept_segments_by_class

# =============================== Transforms =============================== #
def make_test_transform():
    input_size = 224
    return transforms.Compose([
        transforms.Resize((int(input_size/0.875), int(input_size/0.875))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# ======================== Evaluation (multi-framesdir) ==================== #
def forward_logits_10c(model, inputs, model_tag: str):
    if model_tag == "proposed":
        o1, o2, o3, oc, *_ = model(inputs)
        out = o1 + o2 + o3 + oc
    else:
        out = model(inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            for k in ("logits", "out", "cls_logits"):
                if k in out:
                    out = out[k]
                    break
    if not torch.is_tensor(out):
        raise TypeError(f"Model output is not a Tensor: {type(out)}")
    return out.float()

@torch.no_grad()
def evaluate_segments_multi(model,
                            model_tag: str,
                            segments: List[dict],
                            ds_to_model: Dict[str, str],
                            batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tfm = make_test_transform()

    present_model_names: Set[str] = set()
    for seg in segments:
        ds_lab = seg["ds_label"]
        if ds_lab in ds_to_model:
            present_model_names.add(ds_to_model[ds_lab])
    if not present_model_names:
        raise RuntimeError("No overlapping classes between dataset and model taxonomy for selected segments.")

    present_model_names = sorted(list(present_model_names), key=lambda n: CLASS_TO_IDX[n])
    present_model_idx = [CLASS_TO_IDX[n] for n in present_model_names]
    modelidx_to_n = {mi: i for i, mi in enumerate(present_model_idx)}

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[List[float]] = []

    used, skipped_nomap, skipped_noframes = 0, 0, 0

    def load_frame_tensor(fpath: str):
        img = Image.open(fpath).convert("RGB")
        return tfm(img)

    for seg in segments:
        s, e = seg["start"], seg["end"]
        frames_dir = seg["frames_dir"]
        ds_lab = seg["ds_label"]

        if ds_lab not in ds_to_model:
            skipped_nomap += 1
            continue
        model_name = ds_to_model[ds_lab]
        model_idx = CLASS_TO_IDX[model_name]
        n_gt = modelidx_to_n.get(model_idx, None)
        if n_gt is None:
            skipped_nomap += 1
            continue

        frame_paths = []
        for i in range(s, e + 1):
            fpath = os.path.join(frames_dir, FRAME_NAME.format(i))
            if os.path.isfile(fpath):
                frame_paths.append(fpath)
        if len(frame_paths) == 0:
            skipped_noframes += 1
            continue

        logits_sum = None
        bs = batch_size
        for k in range(0, len(frame_paths), bs):
            batch_paths = frame_paths[k:k+bs]
            batch = torch.stack([load_frame_tensor(p) for p in batch_paths], dim=0).to(device, non_blocking=True)
            logits10 = forward_logits_10c(model, batch, model_tag)
            if logits_sum is None:
                logits_sum = logits10.sum(dim=0, keepdim=False)
            else:
                logits_sum += logits10.sum(dim=0, keepdim=False)
        logits_mean = logits_sum / float(len(frame_paths))

        logitsN = torch.stack([logits_mean[mi] for mi in present_model_idx], dim=0)
        probsN = torch.softmax(logitsN, dim=0).cpu().numpy()
        predN = int(probsN.argmax())

        y_true.append(modelidx_to_n[model_idx])
        y_pred.append(predN)
        y_scores.append(probsN.tolist())
        used += 1

    if used == 0:
        raise RuntimeError("No valid segments to evaluate (all skipped).")

    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    n_classes = y_scores.shape[1]

    accuracy = float(accuracy_score(y_true, y_pred) * 100.0)
    f1_micro = float(f1_score(y_true, y_pred, average="micro"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    precision_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        if n_classes >= 3:
            auc_micro = float(roc_auc_score(y_true, y_scores, multi_class='ovr', average='micro'))
            auc_macro = float(roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro'))
        elif n_classes == 2:
            auc_bin = float(roc_auc_score(y_true, y_scores[:, 1]))
            auc_micro = auc_macro = auc_bin
        else:
            auc_micro = auc_macro = 0.0
    except Exception:
        auc_micro = auc_macro = 0.0

    info = {
        "segments_used": used,
        "segments_skipped_nomap": skipped_nomap,
        "segments_skipped_no_frames": skipped_noframes,
        "present_model_names": present_model_names,
    }
    return (accuracy, f1_micro, f1_macro, auc_micro, auc_macro,
            precision_micro, precision_macro, recall_micro, recall_macro, info)

# =============================== Model Loading ============================ #
# Keep only the minimal aliasing needed to load the proposed checkpoint.
import train as tpv5b  # referenced by the pickled model
import utils  # may be referenced by the checkpoint

def _alias_class_into_main(name: str, cls):
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")
        sys.modules["__main__"] = main_mod
    setattr(main_mod, name, cls)

def _prepare_pickle_aliases():
    if hasattr(tpv5b, "Network_Wrapper"):
        _alias_class_into_main("Network_Wrapper", tpv5b.Network_Wrapper)
    if hasattr(tpv5b, "Generator"):
        _alias_class_into_main("Generator", tpv5b.Generator)
    if hasattr(tpv5b, "Features"):
        _alias_class_into_main("Features", tpv5b.Features)
    if hasattr(tpv5b, "NegativeL1Loss"):
        _alias_class_into_main("NegativeL1Loss", tpv5b.NegativeL1Loss)

def load_proposed(device: torch.device):
    _prepare_pickle_aliases()
    model = torch.load(CKPT_PROPOSED, map_location=device)
    return model, "proposed"

# ========================= Sampling helper (random) ====================== #
def sample_per_class_random(kept_segments_by_class: Dict[str, List[dict]],
                            rng: random.Random,
                            per_class: int) -> List[dict]:
    sampled: List[dict] = []
    for cls in IDX_TO_CLASS.values():
        pool = kept_segments_by_class.get(cls, [])
        if len(pool) == 0:
            print(f"[WARN] No available segments for class: {cls}")
            continue
        n_needed = min(per_class, len(pool))
        chosen = rng.sample(pool, n_needed) if n_needed > 0 else []
        if len(chosen) < per_class:
            print(f"[INFO] Class {cls}: only {len(chosen)} available (< {per_class}).")
        sampled.extend(chosen)
    return sampled

# ======================= Metrics aggregation & printing =================== #
METRIC_NAMES = [
    "Acc(%)", "F1_micro", "F1_macro", "AUC_micro", "AUC_macro",
    "Prec_micro", "Prec_macro", "Recall_micro", "Recall_macro"
]

def summarize_mean_std(values_per_run: List[Tuple]) -> List[Tuple[float, float]]:
    cols = list(zip(*values_per_run))
    out = []
    for c in cols:
        m = statistics.mean(c)
        sd = statistics.pstdev(c) if len(c) > 1 else 0.0
        out.append((m, sd))
    return out

def print_mean_std_table(results: Dict[str, List[Tuple]]):
    print("\n=== Summary over runs (mean ± std) ===")
    header = "{:<22s} ".format("Model") + " ".join([f"{n:<12s}" for n in METRIC_NAMES])
    print(header)
    print("-" * len(header))
    for model_name, vals in results.items():
        stats = summarize_mean_std(vals)
        row = "{:<22s} ".format(model_name)
        for (m, sd) in stats:
            row += f"{m:.4f}±{sd:.4f} ".ljust(12)
        print(row)

# ================================ Main =================================== #
def main():
    # 1) Scan JSONs and collect frame-complete single-label segments per class
    kept_segments_by_class = scan_and_collect(ROOT_JSON_DIR)

    print("\n[INFO] Available segments per class after frame check:")
    for i in range(10):
        cls = IDX_TO_CLASS[i]
        print(f"  {cls:25s}: {len(kept_segments_by_class.get(cls, []))}")

    # 2) Load model (proposed only)
    model_specs = [("proposed", load_proposed)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Multiple independent runs with system randomness (no fixed seeds)
    per_model_run_metrics: Dict[str, List[Tuple]] = {name: [] for (name, _) in model_specs}

    for run_idx in range(NUM_RUNS):
        rng = random.Random()  # system entropy; not seeded
        subset = sample_per_class_random(
            kept_segments_by_class,
            rng=rng,
            per_class=SAMPLES_PER_CLASS
        )

        # Report class distribution for this run
        by_class = defaultdict(int)
        for seg in subset:
            by_class[seg["ds_label"]] += 1
        print(f"\n=== Run {run_idx} sampling stats ===")
        for i in range(10):
            cls = IDX_TO_CLASS[i]
            print(f"  {cls:25s}: selected={by_class.get(cls,0):3d}")

        # Evaluate model
        for model_name, loader in model_specs:
            print(f"\n[Run {run_idx}] Evaluating model: {model_name}")
            try:
                model, tag = loader(device)
                res = evaluate_segments_multi(model, tag, subset, DS_TO_MODEL, BATCH_SIZE)
                metrics_tuple = tuple(res[:9])
                info = res[9]
                print(f"[INFO] Used {info['segments_used']} segments | Classes={sorted(list(info['present_model_names']))}")
                print("  Acc(%)={:.4f}, F1_micro={:.4f}, F1_macro={:.4f}, AUC_micro={:.4f}, AUC_macro={:.4f}, "
                      "Prec_micro={:.4f}, Prec_macro={:.4f}, Recall_micro={:.4f}, Recall_macro={:.4f}".format(*metrics_tuple))
                per_model_run_metrics[model_name].append(metrics_tuple)
            except Exception as e:
                print(f"[ERROR] {model_name} failed: {e}")

    # 4) Aggregate mean ± std across runs
    per_model_run_metrics = {k: v for k, v in per_model_run_metrics.items() if len(v) > 0}
    print_mean_std_table(per_model_run_metrics)

if __name__ == "__main__":
    main()