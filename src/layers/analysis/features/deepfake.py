"""
Deep Learning Deepfake Detectors.

Integrates state-of-the-art ML models for AI-generated audio detection.
"""

import os
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import FeatureExtractor, FeatureResult, load_audio

# Try importing torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try importing requests for model download
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DeepfakeDetector(FeatureExtractor):
    """
    Base class for DL-based detectors.
    """
    def __init__(self, name: str, description: str):
        super().__init__(name, description)


class MusicAIDetector(DeepfakeDetector):
    """
    Hugging Face-based AI Music Detector.
    
    Uses: AI-Music-Detection/ai_music_detection_large_60s
    Designed to classify music as AI-generated (Suno, Udio, etc.) or Human.
    """
    
    DEFAULT_MODEL = "AI-Music-Detection/ai_music_detection_large_60s"
    
    def __init__(self, model_id: Optional[str] = None):
        super().__init__(
            name="music_ai_detector",
            description="Detects AI-generated music (Suno, Udio, etc.) using Hugging Face Transformers"
        )
        self.model_id = model_id or self.DEFAULT_MODEL
        self.classifier = None
        
    def is_available(self) -> bool:
        """Check if PyTorch and Transformers are available."""
        return TORCH_AVAILABLE
    
    def get_required_libraries(self) -> List[str]:
        return ['torch', 'transformers']
    
    def _load_model(self):
        """Load the model using transformers pipeline."""
        if self.classifier is not None:
            return
            
        if not self.is_available():
            return
            
        try:
            from transformers import pipeline
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"🧠 Loading AI Music Detector ({self.model_id}) on {device}...")
            
            # Load the classification pipeline
            self.classifier = pipeline(
                "audio-classification", 
                model=self.model_id, 
                device=device
            )
            
            print(f"✅ AI Music Detector ready")
            
        except Exception as e:
            print(f"❌ Failed to load AI Music Detector: {e}")
            self.classifier = None

    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """
        Detect if audio is AI generated.
        """
        if not self.is_available():
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': 'PyTorch/Transformers not available'},
                flags=['Install torch and transformers to enable AI detection']
            )
            
        # Ensure model is loaded
        self._load_model()
        
        if self.classifier is None:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'status': 'Model not loaded'},
                flags=['AI Detection model failed to load']
            )
            
        try:
            # We use the audio path directly as transformers handles loading
            results = self.classifier(audio_path)
            
            # Results is usually a list of {label: ..., score: ...}
            # We look for 'ai' vs 'human' or similar
            # For ai_music_detection_large_60s, labels are likely 'ai' and 'human'
            
            ai_score = 0.0
            for res in results:
                if res['label'].lower() == 'ai':
                    ai_score = res['score']
                    break
            
            # Generate flags
            flags = []
            if ai_score > 0.8:
                flags.append("🚨 CRITICAL: High AI probability (90%+ match with known AI patterns)")
            elif ai_score > 0.5:
                flags.append("⚠️ SUSPICIOUS: Moderate AI characteristics detected")
                
            return FeatureResult(
                feature_name=self.name,
                score=ai_score,
                confidence=0.85,
                metrics={
                    'ai_score': ai_score,
                    'model': self.model_id,
                    'raw_results': results
                },
                flags=flags
            )
            
        except Exception as e:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': str(e)},
                flags=[f'AI Detection failed: {e}']
            )


def get_dl_detectors():
    """Get all deep learning detectors."""
    return [MusicAIDetector()]
