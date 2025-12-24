#!/usr/bin/env python3
"""
FastAPI app for LaMa Inpainting
Accepts an input image and mask, returns inpainted result
"""
import asyncio
import base64
import io
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import environ

import cv2
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo

# Model path (should point to the checkpoint directory)
MODEL_PATH = os.environ.get("LAMA_MODEL_PATH", "./big-lama")
CHECKPOINT_NAME = os.environ.get("LAMA_CHECKPOINT", "best.ckpt")
PAD_MODULO = int(os.environ.get("LAMA_PAD_MODULO", "8"))
# Device selection: 'cuda', 'cpu', or 'auto' (default: auto)
DEVICE = os.environ.get("LAMA_DEVICE", "auto")


# Helper functions
def img_buffer_to_npy_img(img_buffer: bytes) -> np.ndarray:
    """Convert image buffer to numpy array"""
    img_bytes = np.frombuffer(img_buffer, np.uint8)
    img = cv2.imdecode(img_bytes, flags=cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def npy_img_to_base64_png(img: np.ndarray) -> str:
    """Convert numpy image to base64 PNG"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    is_success, png_buffer = cv2.imencode(".png", img_bgr)
    if not is_success:
        raise ValueError("Could not encode image to PNG")
    io_buf = io.BytesIO(png_buffer.tobytes())
    return base64.b64encode(io_buf.getvalue()).decode("utf-8")


def prepare_image_and_mask(image: np.ndarray, mask: np.ndarray, pad_out_to_modulo: int = 8):
    """
    Prepare image and mask for inference

    Args:
        image: RGB image as numpy array (H, W, 3)
        mask: Grayscale mask as numpy array (H, W) or (H, W, 1)
        pad_out_to_modulo: Pad to multiple of this value

    Returns:
        dict with image, mask, and unpad_to_size
    """
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Normalize image to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype('float32') / 255.0

    # Normalize mask to [0, 1]
    if mask.dtype == np.uint8:
        mask = mask.astype('float32') / 255.0

    # Convert to (C, H, W) format
    image = np.transpose(image, (2, 0, 1))
    mask = mask[None, ...]  # Add channel dimension

    # Store original size for unpadding
    orig_height, orig_width = image.shape[1:]

    # Pad to modulo
    if pad_out_to_modulo > 1:
        image = pad_img_to_modulo(image, pad_out_to_modulo)
        mask = pad_img_to_modulo(mask, pad_out_to_modulo)

    return {
        'image': image,
        'mask': mask,
        'unpad_to_size': [orig_height, orig_width]
    }


# Model manager
class LamaInpaintingManager:
    def __init__(self):
        self.model = None
        self.device = None
        self.last_used = None
        self._loading = False
        self.executor = ThreadPoolExecutor(max_workers=3)

    async def load_model(self, model_path: str = MODEL_PATH, checkpoint: str = CHECKPOINT_NAME):
        """Load LaMa model asynchronously"""
        if self.model is not None and not self._loading:
            return  # Already loaded

        if not self._loading:
            self._loading = True
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync, model_path, checkpoint)
                self.last_used = time.time()
            finally:
                self._loading = False

    def _load_model_sync(self, model_path: str, checkpoint: str):
        """Synchronous model loading"""
        print(f"Loading LaMa model from {model_path}...")

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

        # Determine device based on LAMA_DEVICE environment variable
        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif DEVICE == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = torch.device("cuda")
        elif DEVICE == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid LAMA_DEVICE value: {DEVICE}. Must be 'auto', 'cuda', or 'cpu'")

        print(f"Using device: {self.device} (LAMA_DEVICE={DEVICE})")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")

    def unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded")

    async def inpaint(self, image: np.ndarray, mask: np.ndarray, pad_modulo: int = PAD_MODULO) -> np.ndarray:
        """
        Perform inpainting asynchronously

        Args:
            image: RGB image as numpy array (H, W, 3)
            mask: Grayscale mask as numpy array (H, W) or (H, W, 1)
            pad_modulo: Pad to multiple of this value

        Returns:
            Inpainted image as numpy array (H, W, 3)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        self.last_used = time.time()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._inpaint_sync,
            image,
            mask,
            pad_modulo
        )

        self.last_used = time.time()
        return result

    def _inpaint_sync(self, image: np.ndarray, mask: np.ndarray, pad_modulo: int) -> np.ndarray:
        """Synchronous inpainting for thread executor"""
        # Prepare data
        batch_data = prepare_image_and_mask(image, mask, pad_modulo)

        # Convert to tensors and create batch
        batch = {
            'image': torch.from_numpy(batch_data['image']).unsqueeze(0),
            'mask': torch.from_numpy(batch_data['mask']).unsqueeze(0),
        }

        # Extract unpad_to_size (metadata, not a tensor)
        unpad_to_size = batch_data['unpad_to_size']

        # Move to device
        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1

            # Run inference
            batch = self.model(batch)

            # Get result
            result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

            # Unpad to original size
            if unpad_to_size:
                orig_height, orig_width = unpad_to_size
                result = result[:orig_height, :orig_width]

        # Convert to uint8 [0, 255]
        result = np.clip(result * 255, 0, 255).astype('uint8')

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def get_uptime(self) -> float:
        """Get model uptime in seconds"""
        if self.last_used is None:
            return 0
        return time.time() - self.last_used


# Global model manager
lama_manager = LamaInpaintingManager()


# FastAPI app
app = FastAPI(
    title="LaMa Inpainting API",
    description="Image inpainting with LaMa (Large Mask Inpainting)",
    version="1.0.0",
)


# Startup event (Python 3.6 compatible)
@app.on_event("startup")
async def startup_event():
    """Startup event - load model"""
    print("Starting LaMa Inpainting API...")

    # Pre-load model
    try:
        await lama_manager.load_model()
        print("Model pre-loaded successfully")
    except Exception as e:
        print(f"Failed to pre-load model: {e}")
        print(f"Traceback: {traceback.format_exc()}")


# Shutdown event (Python 3.6 compatible)
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - cleanup"""
    lama_manager.unload_model()
    print("Cleanup completed")


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    try:
        await lama_manager.load_model()
        health = lama_manager.model is not None
        status_code = 200 if health else 503
        return JSONResponse(
            content={"status": "healthy" if health else "unhealthy"},
            status_code=status_code,
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(..., description="Input image file"),
    mask: UploadFile = File(..., description="Mask image file (white=inpaint, black=keep)"),
):
    """
    Perform inpainting on the input image using the mask.

    The mask should be a grayscale image where:
    - White pixels (255) indicate areas to inpaint
    - Black pixels (0) indicate areas to keep
    """
    request_start = time.time()

    try:
        # Read image and mask
        io_start = time.time()
        image_buffer = await image.read()
        mask_buffer = await mask.read()
        io_time = time.time() - io_start

        # Decode images
        decode_start = time.time()
        try:
            img_np = img_buffer_to_npy_img(image_buffer)
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Could not decode image: {e}")

        try:
            mask_np = img_buffer_to_npy_img(mask_buffer)
            # Convert to grayscale if needed
            if mask_np.ndim == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Could not decode mask: {e}")
        decode_time = time.time() - decode_start

        # Check dimensions match
        if img_np.shape[:2] != mask_np.shape[:2]:
            raise HTTPException(
                status_code=400,
                detail=f"Image and mask dimensions must match. Image: {img_np.shape[:2]}, Mask: {mask_np.shape[:2]}"
            )

        print(f"Inpainting request - Image size: {img_np.shape[:2][::-1]} (W x H)")

        # Load model
        model_load_start = time.time()
        await lama_manager.load_model()
        model_load_time = time.time() - model_load_start

        # Perform inpainting
        inference_start = time.time()
        result = await lama_manager.inpaint(img_np, mask_np)
        inference_time = time.time() - inference_start

        # Convert to base64
        encode_start = time.time()
        result_base64 = npy_img_to_base64_png(result)
        encode_time = time.time() - encode_start

        total_time = time.time() - request_start

        print(f"⏱️  Timing - Total: {total_time:.3f}s | IO: {io_time:.3f}s | Decode: {decode_time:.3f}s | Model load: {model_load_time:.3f}s | Inference: {inference_time:.3f}s | Encode: {encode_time:.3f}s")

        return JSONResponse(
            content={
                "result": result_base64,
                "format": "png",
                "content_type": "image/png",
                "processing_time": inference_time,
                "total_time": total_time,
                "timing": {
                    "io_read": round(io_time, 3),
                    "decode": round(decode_time, 3),
                    "model_load": round(model_load_time, 3),
                    "inference": round(inference_time, 3),
                    "encode": round(encode_time, 3),
                    "total": round(total_time, 3),
                },
                "input_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
                "output_size": {"width": result.shape[1], "height": result.shape[0]},
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Inpainting failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {e}")


@app.get("/status")
async def get_status():
    """Get current model status"""
    is_loaded = lama_manager.model is not None
    uptime = lama_manager.get_uptime()

    status = {
        "model_loaded": is_loaded,
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "device": str(lama_manager.device) if lama_manager.device else None,
        "last_used": datetime.fromtimestamp(lama_manager.last_used).isoformat()
        if lama_manager.last_used
        else None,
        "model_path": MODEL_PATH,
        "checkpoint": CHECKPOINT_NAME,
    }

    return JSONResponse(content=status)


@app.post("/force_unload")
async def force_unload():
    """Manually unload model"""
    if lama_manager.model is not None:
        lama_manager.unload_model()
        return {"message": "Model unloaded successfully"}
    else:
        return {"message": "Model was not loaded"}


if __name__ == "__main__":
    port = int(environ.get("LAMA_PORT", "8082"))
    log_level = environ.get("LOGGING_LEVEL", "info").lower()
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
    )
