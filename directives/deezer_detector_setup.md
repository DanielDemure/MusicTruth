# Deezer Deepfake Detector - Setup Guide

## Overview

The Deezer Deepfake Detector is based on research from Deezer's AI team:
- **Repository:** https://github.com/deezer/deepfake-detector
- **Paper:** "AI-Generated Music Detection and its Challenges"
- **License:** Research and non-commercial use only

## Installation

### 1. Install PyTorch

```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Additional Dependencies

```bash
pip install requests  # For model download
```

## Usage

The detector will automatically download the model on first use:

```python
from src.layers.analysis.core import Analyzer
from src.config import AnalysisMode

analyzer = Analyzer()
results = analyzer.analyze_audio("song.mp3", mode=AnalysisMode.FORENSIC)

# Deezer score will be in results['features']['deezer_deepfake_detector']
```

## Model Details

- **Input:** 16kHz mono audio
- **Output:** Probability score (0-1)
  - 0.0 = Likely human-made
  - 1.0 = Likely AI-generated
- **Threshold:** 0.5 (configurable)

## Limitations

1. **Research Model:** Not the production model used by Deezer
2. **License:** Non-commercial use only
3. **Dataset Bias:** Trained on specific AI generators (may not generalize)

## Alternative: Custom Model

To use your own model:

1. Replace `MODEL_URL` in `deepfake.py`
2. Implement custom `_preprocess_audio()` if needed
3. Update model architecture in `_load_model()`

## Troubleshooting

### Model Download Fails

```python
# Manual download
wget https://github.com/deezer/deepfake-detector/releases/download/v1.0/model.pth
# Move to: ~/.cache/musictruth/models/deezer_deepfake_v1.pth
```

### CUDA Out of Memory

```python
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Model Not Found

Check cache directory:
```bash
ls ~/.cache/musictruth/models/
```

## Performance

- **CPU:** ~2-5 seconds per 30s clip
- **GPU:** ~0.5-1 second per 30s clip

## References

- [Deezer GitHub](https://github.com/deezer/deepfake-detector)
- [ArXiv Paper](https://arxiv.org/abs/2410.01000)
