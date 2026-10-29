#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm


def load_training_module(path):
    spec = importlib.util.spec_from_file_location('main7_8_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_key(path):
    stem = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def load_2d(path):
    data = np.squeeze(nib.load(str(path)).get_fdata())
    if data.ndim != 2:
        raise RuntimeError(f'Expected 2D NIfTI after squeeze: {path}, shape={data.shape}')
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f'Non-finite values found: {path}')
    return data


def normalize_external(image):
    low = float(image.min())
    high = float(image.max())
    if high <= low:
        raise RuntimeError('Image has no intensity range')
    image = np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    image = 255 - image
    return image


def prepare_input(image_u8, size):
    resized = cv2.resize(image_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array[None, None]), resized


def resize_mask(mask, size):
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def dice_iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    dice = 1.0 if denom == 0 else 2.0 * intersection / denom
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def surface_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() or not target.any():
        return float('nan'), float('nan')
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    target_border = np.logical_xor(target, binary_erosion(target))
    dt_target = distance_transform_edt(~target_border)
    dt_pred = distance_transform_edt(~pred_border)
    distances = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    if distances.size == 0:
        return 0.0, 0.0
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def metrics(pred, target):
    dice, iou = dice_iou(pred, target)
    hd95, assd = surface_metrics(pred, target)
    return {'dice': dice, 'iou': iou, 'hd95': hd95, 'assd': assd}


def load_case(image_path, mask_dir, size):
    case_id = image_path.name[:-7]
    raw = load_2d(image_path)
    image_u8 = normalize_external(raw)
    tensor, display = prepare_input(image_u8, size)
    masks = []
    mask_paths = []
    for annotator in (1, 2, 3):
        mask_path = mask_dir / f'{case_id}_{annotator}.nii.gz'
        if not mask_path.is_file():
            raise FileNotFoundError(f'Missing mask: {mask_path}')
        mask_raw = load_2d(mask_path)
        if mask_raw.shape != raw.shape:
            raise RuntimeError(f'Shape mismatch: {image_path} vs {mask_path}')
        mask = mask_raw == 2
        if not mask.any():
            raise RuntimeError(f'Empty heart mask: {mask_path}')
        masks.append(resize_mask(mask, size))
        mask_paths.append(mask_path)
    votes = sum(mask.astype(np.uint8) for mask in masks)
    majority = votes >= 2
    return tensor, display, masks, majority, mask_paths


def load_models(training_module, model_dir, device, base_ch):
    models = []
    paths = []
    for fold in range(1, 6):
        path = model_dir / f'best_fold{fold}.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Missing model: {path}')
        model = training_module.build_model('resunet', base_ch).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        paths.append(path)
    return models, paths


def predict_probabilities(models, tensor, device):
    tensor = tensor.to(device)
    probabilities = []
    with torch.no_grad():
        for model in models:
            probabilities.append(torch.sigmoid(model(tensor))[0, 0].cpu().numpy())
    return probabilities


def add_panel(ax, image, mask, title, contour_color='red'):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    if mask is not None and mask.any():
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[contour_color], linewidths=1.2)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def create_case_page(pdf, case_id, image, masks, majority, predictions, ensemble, mask_paths, fold_metrics, ensemble_metrics):
    fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27))
    fig.suptitle(f'External case: {case_id}.nii.gz', fontsize=14, fontweight='bold')

    add_panel(axes[0, 0], image, None, f'Input image: {case_id}.nii.gz')
    add_panel(axes[0, 1], image, masks[0], f'Ground truth: {mask_paths[0].name}, label 2', 'lime')
    add_panel(axes[0, 2], image, majority, 'Majority ground truth: annotators 1, 2, 3', 'yellow')

    for fold in range(1, 6):
        row = (fold + 2) // 3
        col = (fold + 2) % 3
        m = fold_metrics[fold - 1]
        add_panel(
            axes[row, col], image, predictions[fold - 1],
            f'Fold {fold}: Dice={m["dice"]:.4f}, HD95={m["hd95"]:.2f}'
        )

    add_panel(
        axes[2, 2], image, ensemble,
        f'Ensemble: Dice={ensemble_metrics["dice"]:.4f}, HD95={ensemble_metrics["hd95"]:.2f}',
        'deepskyblue'
    )
    fig.text(
        0.5, 0.018,
        'Reference for displayed metrics: Annotator 1, label 2 | '
        'Fold predictions: red | Ensemble probability mean threshold: 0.5',
        ha='center', fontsize=8
    )
    plt.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    pdf.savefig(fig, dpi=180)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary_rows = []
    groups = sorted({(row['prediction'], row['reference']) for row in rows})
    for prediction, reference in groups:
        selected = [row for row in rows if row['prediction'] == prediction and row['reference'] == reference]
        item = {'prediction': prediction, 'reference': reference, 'n_cases': len(selected)}
        for name in ('dice', 'iou', 'hd95', 'assd'):
            values = np.array([row[name] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            item[f'{name}_mean'] = float(np.mean(finite)) if finite.size else float('nan')
            item[f'{name}_sd'] = float(np.std(finite, ddof=1)) if finite.size > 1 else float('nan')
            item[f'{name}_median'] = float(np.median(finite)) if finite.size else float('nan')
        summary_rows.append(item)
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-code', type=Path, default=Path('./main7.8.py'))
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--external-images', type=Path, default=Path('./HEART_DATASET/images/testing'))
    parser.add_argument('--external-masks', type=Path, default=Path('./HEART_DATASET/segmentations/testing'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--base-ch', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--expected-cases', type=int, default=49)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_training_module(args.training_code)
    device = module.select_device()
    image_paths = sorted(args.external_images.glob('*.nii.gz'), key=natural_key)
    if len(image_paths) != args.expected_cases:
        raise RuntimeError(f'Expected {args.expected_cases} external cases, found {len(image_paths)}')

    models, model_paths = load_models(module, args.model_dir, device, args.base_ch)
    print(f'Device: {device}', flush=True)
    print(f'External cases: {len(image_paths)}', flush=True)
    print(f'Model directory: {args.model_dir}', flush=True)

    detail_rows = []
    case_rows = []
    pdf_path = args.output_dir / 'external_all_cases.pdf'
    start = time.perf_counter()

    with PdfPages(pdf_path) as pdf:
        bar = tqdm(image_paths, desc='External reevaluation', unit='case', dynamic_ncols=True, file=sys.stderr)
        for index, image_path in enumerate(bar, start=1):
            case_id = image_path.name[:-7]
            tensor, image, masks, majority, mask_paths = load_case(image_path, args.external_masks, args.img_size)
            probabilities = predict_probabilities(models, tensor, device)
            predictions = [prob >= args.threshold for prob in probabilities]
            ensemble_probability = np.mean(np.stack(probabilities, axis=0), axis=0)
            ensemble = ensemble_probability >= args.threshold
            references = {'annotator_1': masks[0], 'annotator_2': masks[1], 'annotator_3': masks[2], 'majority': majority}

            fold_metrics_a1 = []
            for fold, prediction in enumerate(predictions, start=1):
                for reference_name, reference_mask in references.items():
                    value = metrics(prediction, reference_mask)
                    detail_rows.append({
                        'filename': image_path.name,
                        'mask_filename': mask_paths[0].name if reference_name == 'annotator_1' else
                            mask_paths[1].name if reference_name == 'annotator_2' else
                            mask_paths[2].name if reference_name == 'annotator_3' else 'majority_vote',
                        'prediction': f'fold_{fold}',
                        'reference': reference_name,
                        **value,
                        'predicted_area_pixels': int(prediction.sum()),
                        'reference_area_pixels': int(reference_mask.sum()),
                    })
                    if reference_name == 'annotator_1':
                        fold_metrics_a1.append(value)

            ensemble_by_reference = {}
            for reference_name, reference_mask in references.items():
                value = metrics(ensemble, reference_mask)
                ensemble_by_reference[reference_name] = value
                detail_rows.append({
                    'filename': image_path.name,
                    'mask_filename': mask_paths[0].name if reference_name == 'annotator_1' else
                        mask_paths[1].name if reference_name == 'annotator_2' else
                        mask_paths[2].name if reference_name == 'annotator_3' else 'majority_vote',
                    'prediction': 'ensemble',
                    'reference': reference_name,
                    **value,
                    'predicted_area_pixels': int(ensemble.sum()),
                    'reference_area_pixels': int(reference_mask.sum()),
                })

            create_case_page(
                pdf, case_id, image, masks, majority, predictions, ensemble,
                mask_paths, fold_metrics_a1, ensemble_by_reference['annotator_1']
            )
            case_rows.append({
                'filename': image_path.name,
                'annotator_1_mask': mask_paths[0].name,
                'annotator_2_mask': mask_paths[1].name,
                'annotator_3_mask': mask_paths[2].name,
                'fold_dice_mean_vs_annotator_1': float(np.mean([m['dice'] for m in fold_metrics_a1])),
                'fold_dice_sd_vs_annotator_1': float(np.std([m['dice'] for m in fold_metrics_a1], ddof=1)),
                'ensemble_dice_vs_annotator_1': ensemble_by_reference['annotator_1']['dice'],
                'ensemble_dice_vs_annotator_2': ensemble_by_reference['annotator_2']['dice'],
                'ensemble_dice_vs_annotator_3': ensemble_by_reference['annotator_3']['dice'],
                'ensemble_dice_vs_majority': ensemble_by_reference['majority']['dice'],
                'ensemble_hd95_vs_annotator_1': ensemble_by_reference['annotator_1']['hd95'],
                'ensemble_assd_vs_annotator_1': ensemble_by_reference['annotator_1']['assd'],
            })
            elapsed = time.perf_counter() - start
            remaining = elapsed / index * (len(image_paths) - index)
            print(
                f'[Progress] case={index}/{len(image_paths)} filename={image_path.name} '
                f'ensemble_dice={ensemble_by_reference["annotator_1"]["dice"]:.4f} '
                f'elapsed={time.strftime("%H:%M:%S", time.gmtime(elapsed))} '
                f'estimated_remaining={time.strftime("%H:%M:%S", time.gmtime(remaining))}',
                flush=True,
            )

    write_csv(args.output_dir / 'external_metrics_detailed.csv', detail_rows)
    write_csv(args.output_dir / 'external_metrics_by_case.csv', case_rows)
    summary_rows = summarize(detail_rows)
    write_csv(args.output_dir / 'external_metrics_summary.csv', summary_rows)

    config = {
        'training_code': str(args.training_code),
        'model_dir': str(args.model_dir),
        'model_paths': [str(path) for path in model_paths],
        'external_images': str(args.external_images),
        'external_masks': str(args.external_masks),
        'output_dir': str(args.output_dir),
        'device': str(device),
        'img_size': args.img_size,
        'base_ch': args.base_ch,
        'threshold': args.threshold,
        'expected_cases': args.expected_cases,
        'ensemble_method': 'mean_probability_then_threshold',
        'heart_label': 2,
        'external_invert': True,
    }
    with (args.output_dir / 'reevaluation_config.json').open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

    print(f'PDF: {pdf_path}', flush=True)
    print(f'Detailed CSV: {args.output_dir / "external_metrics_detailed.csv"}', flush=True)
    print(f'Case CSV: {args.output_dir / "external_metrics_by_case.csv"}', flush=True)
    print(f'Summary CSV: {args.output_dir / "external_metrics_summary.csv"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
