#!/bin/bash

# LaMa Inpainting FastAPI Server
# Usage: ./fastapi.sh [options]
#
# Examples:
#   ./fastapi.sh                    # Start with GPU (auto)
#   ./fastapi.sh cpu                # Force CPU mode
#   ./fastapi.sh cuda               # Force GPU mode
#   ./fastapi.sh cuda 8888          # GPU mode on port 8888

# Configuration
DEVICE="${1:-auto}"  # Default: auto (use GPU if available)
PORT="${2:-8082}"    # Default port: 8082

# Model configuration (can be overridden)
export LAMA_MODEL_PATH="${LAMA_MODEL_PATH:-./big-lama}"
export LAMA_CHECKPOINT="${LAMA_CHECKPOINT:-best.ckpt}"
export LAMA_PAD_MODULO="${LAMA_PAD_MODULO:-8}"
export LAMA_DEVICE="${DEVICE}"
export LAMA_PORT="${PORT}"

# Logging configuration
export LOGGING_LEVEL="${LOGGING_LEVEL:-info}"

# Print configuration
echo "=================================================="
echo "🦙 LaMa Inpainting FastAPI Server"
echo "=================================================="
echo "Configuration:"
echo "  Model Path:     ${LAMA_MODEL_PATH}"
echo "  Checkpoint:     ${LAMA_CHECKPOINT}"
echo "  Device:         ${LAMA_DEVICE}"
echo "  Port:           ${LAMA_PORT}"
echo "  Log Level:      ${LOGGING_LEVEL}"
echo "=================================================="
echo ""

# Optional: Kill existing processes
# Uncomment the line below to auto-kill existing server
# pkill -f "uvicorn.*fastapi_app:app"

# Start uvicorn server
uvicorn fastapi_app:app \
  --host 0.0.0.0 \
  --port "${LAMA_PORT}" \
  --workers 1 \
  --log-level "${LOGGING_LEVEL}" \
  --no-access-log
