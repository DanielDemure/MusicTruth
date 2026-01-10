"""
Structural feature extractors for AI music detection.

Analyzes song structure, repetition patterns, and self-similarity.
"""

import numpy as np
import librosa
from typing import Optional

from .base import FeatureExtractor, FeatureResult, load_audio

class StructuralComplexityAnalyzer(FeatureExtractor):
    """
    Analyzes structural complexity using Self-Similarity Matrices.
    
    AI music often has high local repetition but lacks long-term
    structural coherence (e.g., A-B-A-C structure).
    """
    
    def __init__(self):
        super().__init__(
            name="structural_analysis",
            description="Analyzes structural coherence and repetition"
        )
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
            
        # 1. Compute Chromagram (robust to timbre changes)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        
        # 2. Apply recurrence quantification analysis (RQA) via Recurrence Matrix
        # This is expensive for long tracks, need to sub-sample or use standard SSM
        # Use librosa.segment.recurrence_matrix
        
        # Downsample chroma for speed
        chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
        
        rec = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity', sym=True)
        
        # 3. Detect structure using Novelty Curve
        # Kernel checkerboard
        # Or simply use simple metrics from the matrix
        
        # Repetition score: mean of off-diagonal
        # Structure score: presence of blocks?
        
        density = np.mean(rec)
        
        # AI often loops -> High density of recurrence?
        # Or chaotic -> Low density?
        
        # This feature is interpretative. 
        # High score for "Too Repetitive" (Loop generation)
        # High score for "Zero Repetition" (Random generation)
        
        score = 0.0
        flags = []
        
        if density > 0.4:
            score = 0.5
            flags.append(f"High structural repetition (Density: {density:.2f})")
            
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.4,
            metrics={
                'recurrence_density': float(density)
            },
            flags=flags
        )

def get_structural_extractors():
    return [StructuralComplexityAnalyzer()]
