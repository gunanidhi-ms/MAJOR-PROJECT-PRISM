#!/usr/bin/env bash
# ============================================================
# PRISM Installation Script for macOS & Linux
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================="
echo -e "       PRISM Installation Script         "
echo -e "=========================================${NC}"
echo ""

OS="$(uname -s)"
case "$OS" in
  Darwin*) PLATFORM="macOS" ;;
  Linux*)  PLATFORM="Linux" ;;
  *)       echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1 ;;
esac
echo -e "${BOLD}Platform: ${PLATFORM}${NC}"
echo ""

PRISM_MODEL="llama3.2:3b"

# ── [1/6] Check Python ────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking Python installation...${NC}"

PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
    MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)")
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
      PYTHON="$cmd"
      VERSION=$("$cmd" --version 2>&1)
      echo -e "${GREEN}Python found: $VERSION${NC}"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo -e "${RED}Python 3.9+ is required but not found.${NC}"
  [ "$PLATFORM" = "macOS" ] && echo -e "  Install: ${CYAN}brew install python3${NC}"
  [ "$PLATFORM" = "Linux"  ] && echo -e "  Install: ${CYAN}sudo apt install python3 python3-venv python3-pip${NC}"
  exit 1
fi

# ── [2/6] Virtual Environment ─────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Setting up virtual environment...${NC}"

if [ ! -d "venv" ]; then
  echo "Creating new virtual environment 'venv'..."
  $PYTHON -m venv venv
  echo -e "${GREEN}Virtual environment created.${NC}"
else
  echo -e "${GREEN}Virtual environment 'venv' already exists.${NC}"
fi

# shellcheck disable=SC1091
source venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# ── [3/6] GPU Detection + PyTorch ────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Detecting GPU for PyTorch / TotalSegmentator acceleration...${NC}"

GPU_DETECTED=false
CUDA_AVAILABLE=false

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)
  if [ -n "$GPU_INFO" ]; then
    echo -e "${GREEN}NVIDIA GPU detected: $GPU_INFO${NC}"
    GPU_DETECTED=true
    CUDA_AVAILABLE=true
  fi
fi

if [ "$PLATFORM" = "macOS" ]; then
  ARCH=$(uname -m)
  if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}Apple Silicon (M-series) detected - PyTorch will use MPS backend.${NC}"
    GPU_DETECTED=true
  fi
fi

if [ "$CUDA_AVAILABLE" = true ]; then
  echo -e "${GREEN}Attempting CUDA PyTorch installation (cu124)...${NC}"
  if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; then
    echo -e "${GREEN}CUDA PyTorch installed successfully.${NC}"
  else
    echo -e "${YELLOW}CUDA wheel not available for this Python version. Installing standard PyTorch...${NC}"
    pip install torch torchvision
  fi
elif [ "$PLATFORM" = "macOS" ] && [ "$(uname -m)" = "arm64" ]; then
  echo -e "${GREEN}Installing PyTorch with MPS support for Apple Silicon...${NC}"
  pip install torch torchvision
  echo -e "${GREEN}MPS PyTorch installed.${NC}"
else
  echo -e "${YELLOW}No NVIDIA GPU detected. Installing standard CPU PyTorch...${NC}"
  pip install torch torchvision
  echo -e "${GREEN}CPU PyTorch installed.${NC}"
fi

# ── [4/6] Python Dependencies ─────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
  echo "Installing from root requirements.txt..."
  pip install -r requirements.txt
  echo -e "${GREEN}Python dependencies installed.${NC}"
else
  echo -e "${YELLOW}Root requirements.txt not found. Installing per-phase...${NC}"
  for phase_req in \
    "phase1_ingestion/requirements.txt" \
    "phase2_segmentation/requirements.txt" \
    "phase3_reporting/requirements.txt"; do
    [ -f "$phase_req" ] && pip install -r "$phase_req"
  done
fi

# ── [5/6] Ollama ──────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Setting up Ollama (AI Report Generation)...${NC}"

if ! command -v ollama &>/dev/null; then
  echo -e "${YELLOW}Ollama is not installed. Attempting to install...${NC}"
  if [ "$PLATFORM" = "macOS" ] && command -v brew &>/dev/null; then
    brew install ollama
  else
    curl -fsSL https://ollama.com/install.sh | sh || true
  fi
fi

if command -v ollama &>/dev/null; then
  # Remove incompatible multimodal model if present
  if ollama list 2>/dev/null | grep -q "llama3.2-vision"; then
    echo -e "${YELLOW}Removing incompatible llama3.2-vision model...${NC}"
    ollama rm llama3.2-vision 2>/dev/null || true
    echo -e "${GREEN}llama3.2-vision removed.${NC}"
  fi

  # Pull or verify model
  if ollama list 2>/dev/null | grep -q -F "$PRISM_MODEL"; then
    echo -e "${GREEN}Model '$PRISM_MODEL' is ready.${NC}"
  else
    echo -e "${YELLOW}Pulling '$PRISM_MODEL' (2.0 GB - recommended for PRISM)...${NC}"
    ollama pull "$PRISM_MODEL" || echo -e "${YELLOW}Pull failed - PRISM will use template fallback.${NC}"
  fi
else
  echo -e "${YELLOW}Ollama is not installed. AI report generation will use template fallback.${NC}"
  echo -e "Install from: ${CYAN}https://ollama.com/${NC}"
fi

# ── [6/6] Frontend (Node.js) ──────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Installing frontend dependencies...${NC}"

if ! command -v npm &>/dev/null; then
  echo -e "${RED}npm is not installed. Please install Node.js (v18+) from https://nodejs.org/${NC}"
  echo -e "${YELLOW}Frontend will not be able to run without Node.js.${NC}"
else
  if [ -d "prism-react" ]; then
    (cd prism-react && npm install)
    echo -e "${GREEN}Frontend dependencies installed.${NC}"
  else
    echo -e "${RED}prism-react directory not found.${NC}"
  fi
fi

# ── Complete ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}========================================="
echo -e "       PRISM Installation Complete       "
echo -e "=========================================${NC}"
echo ""
if [ "$CUDA_AVAILABLE" = true ]; then
  echo -e "GPU acceleration: ${GREEN}ENABLED (Ollama GPU offload + NVIDIA GPU detected)${NC}"
elif [ "$GPU_DETECTED" = true ]; then
  echo -e "GPU acceleration: ${GREEN}ENABLED (Apple MPS backend)${NC}"
else
  echo -e "GPU acceleration: ${YELLOW}CPU mode (no NVIDIA GPU found)${NC}"
fi
echo -e "LLM model:        ${GREEN}$PRISM_MODEL${NC}"
echo ""
echo -e "You can now run PRISM by executing: ${GREEN}./start_prism.sh${NC}"
echo ""
