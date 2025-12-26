# LaMa Inpainting Service

FastAPI web service for LaMa (Large Mask Inpainting) with Docker support.

## Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Build and start
docker-compose up -d

# Check status
curl http://localhost:8082/ping

# View logs
docker-compose logs -f
```

### Option 2: Direct Host Deployment

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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/status` | GET | Model status and info |
| `/inpaint` | POST | Perform inpainting |
| `/force_unload` | POST | Unload model from memory |

**Inpaint Request:**
```bash
curl -X POST http://localhost:8082/inpaint \
  -F "image=@image.jpg" \
  -F "mask=@mask.png" \
  > response.json
```

**Mask Format:** White pixels (255) = inpaint, Black pixels (0) = keep

### Option 3: CLI Inference (No Server)

```bash
# GPU (auto-detect)
python inference_gpu.py -i image.jpg -m mask.png -o result.png

# CPU only
python inference_cpu.py -i image.jpg -m mask.png -o result.png
```

## Docker Configuration

**Environment Variables:**
- `LAMA_MODEL_PATH`: Model directory (default: `/opt/ml/model/big-lama`)
- `LAMA_CHECKPOINT`: Checkpoint file (default: `best.ckpt`)
- `LAMA_DEVICE`: `auto`, `cuda`, or `cpu` (default: `auto`)
- `LAMA_PORT`: Internal port (default: `8080`)
- `LOGGING_LEVEL`: Log level (default: `info`)

**Change GPU:**
```yaml
# In docker-compose.yml
device_ids: ['1']  # Use GPU 1 instead of GPU 0
```

**CPU-only mode:**
```yaml
# In docker-compose.yml
environment:
  - LAMA_DEVICE=cpu
# Remove the deploy section
```

## Dependencies

**Included in `requirements-docker.txt`:**
- PyTorch 1.11.0 + CUDA 11.3 (from base image)
- opencv-python, numpy, Pillow
- fastapi, uvicorn, python-multipart
- omegaconf, hydra-core
- pytorch-lightning, kornia
- scikit-image, albumentations

## Troubleshooting

**Port in use:** Change `8082:8080` to `8083:8080` in docker-compose.yml

**GPU not found:** Install nvidia-docker2 and restart Docker

**Model not found:** Ensure `./big-lama/config.yaml` and `./big-lama/models/best.ckpt` exist

**View logs:** `docker-compose logs -f lama-inpainting`
