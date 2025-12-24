# LaMa Inpainting FastAPI

A FastAPI web service for LaMa (Large Mask Inpainting) that accepts an input image and mask, and returns the inpainted result.

## Overview

This implementation provides three ways to use LaMa inpainting:

1. **FastAPI Web Service** (`fastapi_app.py`) - REST API for remote inference
2. **Local GPU Inference Script** (`inference_gpu.py`) - Direct Python inference with GPU support
3. **Local CPU Inference Script** (`inference_cpu.py`) - Direct Python inference on CPU only

## Installation

Make sure you have all the required dependencies installed:

```bash
pip install fastapi uvicorn python-multipart requests
```

The LaMa model dependencies should already be installed if you can run the original `bin/predict.py`.

## Model Setup

Download and prepare the LaMa model:

```bash
# The model should be in the big-lama directory (or set LAMA_MODEL_PATH)
# Structure should be:
# big-lama/
#   ├── config.yaml
#   └── models/
#       └── best.ckpt
```

## Usage

### Option 1: FastAPI Web Service

#### Start the Server

**Option A: Using the shell script (Recommended)**

```bash
# Start with auto GPU detection (default)
bash fastapi.sh

# Force CPU mode
bash fastapi.sh cpu

# Force GPU mode
bash fastapi.sh cuda

# Custom port
bash fastapi.sh cuda 8888
```

**Option B: Using Python directly**

```bash
# Using default settings (model at ./big-lama, port 8082, auto GPU)
python fastapi_app.py

# With custom settings
LAMA_MODEL_PATH=./big-lama LAMA_DEVICE=cuda LAMA_PORT=8082 python fastapi_app.py

# Force CPU mode
LAMA_DEVICE=cpu python fastapi_app.py
```

Environment variables:
- `LAMA_MODEL_PATH`: Path to model directory (default: `./big-lama`)
- `LAMA_CHECKPOINT`: Checkpoint filename (default: `best.ckpt`)
- `LAMA_DEVICE`: Device selection: `auto` (default), `cuda`, or `cpu`
- `LAMA_PORT`: Server port (default: `8082`)
- `LAMA_PAD_MODULO`: Padding modulo value (default: `8`)
- `LOGGING_LEVEL`: Log level (default: `info`)

#### API Endpoints

##### `GET /ping`
Health check endpoint.

```bash
curl http://localhost:8082/ping
```

Response:
```json
{"status": "healthy"}
```

##### `GET /status`
Get model status and information.

```bash
curl http://localhost:8082/status
```

Response:
```json
{
  "model_loaded": true,
  "uptime_seconds": 123.45,
  "uptime_hours": 0.034,
  "device": "cuda",
  "last_used": "2024-01-01T12:00:00",
  "model_path": "./big-lama",
  "checkpoint": "best.ckpt"
}
```

##### `POST /inpaint`
Perform inpainting on an image.

```bash
curl -X POST http://localhost:8082/inpaint \
  -F "image=@path/to/image.jpg" \
  -F "mask=@path/to/mask.png" \
  > response.json
```

**Request:**
- `image`: Input image file (multipart/form-data)
- `mask`: Mask image file (multipart/form-data)
  - White pixels (255) = areas to inpaint
  - Black pixels (0) = areas to keep

**Response:**
```json
{
  "result": "base64_encoded_png_image",
  "format": "png",
  "content_type": "image/png",
  "processing_time": 1.234,
  "total_time": 1.456,
  "timing": {
    "io_read": 0.012,
    "decode": 0.034,
    "model_load": 0.001,
    "inference": 1.234,
    "encode": 0.175,
    "total": 1.456
  },
  "input_size": {"width": 512, "height": 512},
  "output_size": {"width": 512, "height": 512}
}
```

##### `POST /force_unload`
Manually unload the model from memory (for debugging).

```bash
curl -X POST http://localhost:8082/force_unload
```

### Option 2: Local Inference Scripts

Run inpainting directly without starting the API server:

#### GPU Inference (`inference_gpu.py`)

Uses GPU if available, falls back to CPU automatically:

```bash
# Basic usage
python inference_gpu.py --input image.jpg --mask mask.png --output result.png

# Short form
python inference_gpu.py -i image.jpg -m mask.png -o result.png

# With custom model path
python inference_gpu.py -i image.jpg -m mask.png -o result.png \
  --model-path ./big-lama

# With custom padding
python inference_gpu.py -i image.jpg -m mask.png -o result.png \
  --pad-modulo 8
```

#### CPU-Only Inference (`inference_cpu.py`)

Forces CPU mode (useful for testing or when GPU is not needed):

```bash
# Basic usage
python inference_cpu.py --input image.jpg --mask mask.png --output result.png

# Short form
python inference_cpu.py -i image.jpg -m mask.png -o result.png
```

Options:
- `-i, --input`: Path to input image file (required)
- `-m, --mask`: Path to mask image file (required)
- `-o, --output`: Path to output image file (required)
- `--model-path`: Path to model directory (default: `./big-lama`)
- `--checkpoint`: Checkpoint filename (default: `best.ckpt`)
- `--pad-modulo`: Pad to multiple of this value (default: `8`)

## Python Client Example

```python
import requests
import base64
import cv2
import numpy as np

# API endpoint
api_url = "http://localhost:8082"

# Load images
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
with open("mask.png", "rb") as f:
    mask_bytes = f.read()

# Send request
files = {
    'image': ('image.jpg', image_bytes, 'image/jpeg'),
    'mask': ('mask.png', mask_bytes, 'image/png'),
}
response = requests.post(f"{api_url}/inpaint", files=files)

# Get result
if response.status_code == 200:
    result = response.json()

    # Decode base64 image
    img_bytes = base64.b64decode(result['result'])
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save result
    cv2.imwrite("result.png", img)

    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Timing breakdown: {result['timing']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Mask Format

The mask image should be a grayscale image where:
- **White pixels (value 255)**: Areas to **inpaint** (fill in)
- **Black pixels (value 0)**: Areas to **keep** (preserve)

The mask should have the same dimensions as the input image.

## Performance Tips

1. **GPU Usage**: The model will automatically use GPU if available (CUDA)
2. **Memory**: The model stays loaded in memory for faster subsequent requests
3. **Padding**: Images are automatically padded to multiples of 8 (default) for model compatibility
4. **Batch Size**: Currently processes one image at a time

## Comparison with Original `bin/predict.py`

| Feature | `bin/predict.py` | FastAPI (`fastapi_app.py`) | GPU (`inference_gpu.py`) | CPU (`inference_cpu.py`) |
|---------|------------------|----------------------------|--------------------------|--------------------------|
| Input | Folder only | Single image + mask | Single image + mask | Single image + mask |
| Interface | CLI | REST API | CLI | CLI |
| Device | CPU | Auto/GPU/CPU | Auto (GPU preferred) | CPU only |
| Use case | Batch processing | Web service | Quick local inference | CPU-only inference |
| Remote access | No | Yes | No | No |
| Persistent model | No | Yes | No | No |

## Troubleshooting

### Model not found
```
FileNotFoundError: Config not found: ./big-lama/config.yaml
```
**Solution**: Set the correct model path using `LAMA_MODEL_PATH` environment variable or `--model-path` argument.

### CUDA out of memory
**Solution**: The model will try to recover automatically. If it persists, use a smaller image or call `/force_unload` to free memory.

### Image and mask dimensions don't match
```
HTTPException: Image and mask dimensions must match
```
**Solution**: Ensure both images have the same width and height.

## License

Same as the original LaMa project.
