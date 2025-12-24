#!/usr/bin/env python3
"""
Local LaMa Inpainting inference script.

This script performs inpainting inference directly without requiring
the FastAPI server to be running. It loads the model locally and processes
images directly.

Usage:
    python inference_gpu.py --input image.jpg --mask mask.png --output result.png
    python inference_gpu.py -i image.jpg -m mask.png -o result.png --model-path ./big-lama
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Set threading environment variables BEFORE importing torch (same as bin/predict.py)
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

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo

# Default paths
DEFAULT_MODEL_PATH = os.environ.get("LAMA_MODEL_PATH", "./big-lama")
DEFAULT_CHECKPOINT = os.environ.get("LAMA_CHECKPOINT", "best.ckpt")


def load_model(model_path: str = DEFAULT_MODEL_PATH, checkpoint: str = DEFAULT_CHECKPOINT):
    """Load LaMa model"""
    print(f"🔄 Loading LaMa model from: {model_path}")

    # Load training config
    train_config_path = os.path.join(model_path, 'config.yaml')
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Config not found: {train_config_path}")

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # Load checkpoint
    checkpoint_path = os.path.join(model_path, 'models', checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    print(f"✅ Model loaded successfully")
    return model, device


def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"📸 Loading image: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"📏 Image size: {img.shape[1]} x {img.shape[0]} (W x H)")
    return img


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask from file path"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    print(f"🎭 Loading mask: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode mask: {mask_path}")

    print(f"📏 Mask size: {mask.shape[1]} x {mask.shape[0]} (W x H)")
    return mask


def prepare_batch(image: np.ndarray, mask: np.ndarray, pad_modulo: int = 8):
    """
    Prepare image and mask for inference

    Args:
        image: RGB image (H, W, 3) uint8
        mask: Grayscale mask (H, W) uint8
        pad_modulo: Pad to multiple of this value

    Returns:
        Batch dict ready for model
    """
    # Check dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions must match. "
            f"Image: {image.shape[:2]}, Mask: {mask.shape[:2]}"
        )

    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    mask = mask.astype('float32') / 255.0

    # Convert to (C, H, W) format
    image = np.transpose(image, (2, 0, 1))
    mask = mask[None, ...]  # Add channel dimension

    # Store original size
    orig_height, orig_width = image.shape[1:]

    # Pad to modulo
    if pad_modulo > 1:
        image = pad_img_to_modulo(image, pad_modulo)
        mask = pad_img_to_modulo(mask, pad_modulo)

    # Create batch
    batch = {
        'image': torch.from_numpy(image).unsqueeze(0),
        'mask': torch.from_numpy(mask).unsqueeze(0),
        'unpad_to_size': [orig_height, orig_width]
    }

    return batch


def perform_inpainting(
    model,
    device,
    image: np.ndarray,
    mask: np.ndarray,
    pad_modulo: int = 8
) -> np.ndarray:
    """
    Perform inpainting inference

    Args:
        model: Loaded LaMa model
        device: torch device
        image: RGB image (H, W, 3) uint8
        mask: Grayscale mask (H, W) uint8
        pad_modulo: Pad to multiple of this value

    Returns:
        Inpainted image (H, W, 3) uint8
    """
    print(f"🚀 Starting inpainting...")

    start_time = time.time()

    # Prepare batch
    batch = prepare_batch(image, mask, pad_modulo)

    # Extract unpad_to_size before moving to device (it's metadata, not a tensor)
    orig_height, orig_width = batch.pop('unpad_to_size')

    # Move to device and run inference
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        # Run model
        batch = model(batch)

        # Get result
        result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        # Unpad to original size
        result = result[:orig_height, :orig_width]

    # Convert to uint8 [0, 255]
    result = np.clip(result * 255, 0, 255).astype('uint8')

    processing_time = time.time() - start_time
    print(f"⏱️  Processing completed in {processing_time:.2f} seconds")

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_image(img: np.ndarray, output_path: str):
    """Save image to file"""
    print(f"💾 Saving image: {output_path}")

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Get file extension
    ext = Path(output_path).suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        cv2.imwrite(output_path, img_bgr)

    print("✅ Image saved successfully")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Perform LaMa inpainting inference locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input photo.jpg --mask mask.png --output result.png
  %(prog)s -i photo.jpg -m mask.png -o result.png
  %(prog)s -i photo.jpg -m mask.png -o result.png --model-path ./big-lama
  %(prog)s -i photo.jpg -m mask.png -o result.png --pad-modulo 8

Note: The mask should be a grayscale image where:
  - White pixels (255) indicate areas to inpaint
  - Black pixels (0) indicate areas to keep
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input image file"
    )

    parser.add_argument(
        "--mask", "-m",
        type=str,
        required=True,
        help="Path to mask image file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output image file"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model directory (default: {DEFAULT_MODEL_PATH})"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint filename (default: {DEFAULT_CHECKPOINT})"
    )

    parser.add_argument(
        "--pad-modulo",
        type=int,
        default=8,
        help="Pad image to multiple of this value (default: 8)"
    )

    return parser.parse_args()


def main():
    """Main inference function"""
    print("🧠 LaMa Local Inpainting Inference")
    print("=" * 40)

    total_start = time.time()
    args = parse_arguments()

    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model_start = time.time()
        model, device = load_model(args.model_path, args.checkpoint)
        model_time = time.time() - model_start

        # Load input image and mask
        load_start = time.time()
        input_img = load_image(args.input)
        input_mask = load_mask(args.mask)
        load_time = time.time() - load_start

        # Perform inpainting
        inference_start = time.time()
        output_img = perform_inpainting(model, device, input_img, input_mask, args.pad_modulo)
        inference_time = time.time() - inference_start

        # Save output image
        save_start = time.time()
        save_image(output_img, args.output)
        save_time = time.time() - save_start

        total_time = time.time() - total_start

        print("\n🎉 Inpainting completed successfully!")
        print(f"\n⏱️  Timing Summary:")
        print(f"   Model loading:  {model_time:.2f}s")
        print(f"   Image loading:  {load_time:.2f}s")
        print(f"   Inference:      {inference_time:.2f}s")
        print(f"   Saving:         {save_time:.2f}s")
        print(f"   ──────────────────────────")
        print(f"   Total time:     {total_time:.2f}s")

    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
