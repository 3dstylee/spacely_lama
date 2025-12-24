#!/usr/bin/env python3
"""
Simple LaMa inpainting - wraps bin/predict.py logic for single images
Based directly on bin/predict.py
"""
import os
import sys
import argparse
import time

# Set threading env vars BEFORE importing torch (same as bin/predict.py)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo


def load_image_as_tensor(image_path, pad_out_to_modulo=8):
    """Load image and convert to tensor"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))

    orig_height, orig_width = img.shape[1:]
    if pad_out_to_modulo > 1:
        img = pad_img_to_modulo(img, pad_out_to_modulo)

    return img, (orig_height, orig_width)


def load_mask_as_tensor(mask_path, pad_out_to_modulo=8):
    """Load mask and convert to tensor"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype('float32') / 255.0
    mask = mask[None, ...]

    if pad_out_to_modulo > 1:
        mask = pad_img_to_modulo(mask, pad_out_to_modulo)

    return mask


def main():
    total_start = time.time()

    parser = argparse.ArgumentParser(description='Simple LaMa Inpainting')
    parser.add_argument('-i', '--input', required=True, help='Input image')
    parser.add_argument('-m', '--mask', required=True, help='Mask image')
    parser.add_argument('-o', '--output', required=True, help='Output image')
    parser.add_argument('--model-path', default='./big-lama', help='Model path')
    parser.add_argument('--checkpoint', default='best.ckpt', help='Checkpoint name')
    args = parser.parse_args()

    # Model loading
    print(f"Loading model from {args.model_path}...")
    model_start = time.time()

    # Load config (same as bin/predict.py)
    train_config_path = os.path.join(args.model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # Load model (same as bin/predict.py lines 44, 58-61)
    device = torch.device("cpu")  # Hardcoded to CPU like original
    checkpoint_path = os.path.join(args.model_path, 'models', args.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    model_time = time.time() - model_start
    print(f"✓ Model loaded in {model_time:.2f}s")
    print(f"  Device: {device}")

    # Load images
    print(f"Loading images...")
    load_start = time.time()
    image, unpad_to_size = load_image_as_tensor(args.input, pad_out_to_modulo=8)
    mask = load_mask_as_tensor(args.mask, pad_out_to_modulo=8)

    # Create batch (same as bin/predict.py line 74)
    batch = {
        'image': torch.from_numpy(image).unsqueeze(0),
        'mask': torch.from_numpy(mask).unsqueeze(0),
    }
    load_time = time.time() - load_start
    print(f"✓ Images loaded in {load_time:.2f}s")

    # Inference
    print("Running inference...")
    inference_start = time.time()

    # Run inference (same as bin/predict.py lines 82-90)
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)

        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        # Unpad
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    inference_time = time.time() - inference_start
    print(f"✓ Inference completed in {inference_time:.2f}s")

    # Save (same as bin/predict.py lines 92-94)
    save_start = time.time()
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, cur_res)
    save_time = time.time() - save_start

    total_time = time.time() - total_start

    print(f"\n{'='*50}")
    print(f"✅ Saved result to {args.output}")
    print(f"\n⏱️  Timing Summary:")
    print(f"   Model loading:  {model_time:.2f}s")
    print(f"   Image loading:  {load_time:.2f}s")
    print(f"   Inference:      {inference_time:.2f}s")
    print(f"   Saving:         {save_time:.2f}s")
    print(f"   ──────────────────────────")
    print(f"   Total time:     {total_time:.2f}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
