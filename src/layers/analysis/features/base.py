"""
Base classes for feature extraction in MusicTruth.

Provides abstract base class and common utilities for all feature extractors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class FeatureResult:
    """
    Standardized result from a feature extractor.
    
    Attributes:
        feature_name: Name of the feature
        score: AI suspicion score (0-1, higher = more suspicious)
        confidence: Confidence in the score (0-1)
        metrics: Raw metric values
        flags: List of human-readable findings
        metadata: Additional information
    """
    feature_name: str
    score: float = 0.0
    confidence: float = 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_name': self.feature_name,
            'score': float(self.score),
            'confidence': float(self.confidence),
            'metrics': self._serialize_metrics(self.metrics),
            'flags': self.flags,
            'metadata': self.metadata
        }
    
    @staticmethod
    def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to Python types for JSON serialization."""
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized


class FeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    Each feature extractor should:
    1. Implement extract() to compute features from audio
    2. Implement score() to convert features to AI suspicion score
    3. Optionally override is_available() to check dependencies
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize feature extractor.
        
        Args:
            name: Unique name for this feature
            description: Human-readable description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None, 
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            y: Pre-loaded audio samples (optional, for efficiency)
            sr: Sample rate (optional, for efficiency)
            **kwargs: Additional parameters
        
        Returns:
            FeatureResult with computed metrics and score
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if this feature extractor can run (dependencies available).
        
        Returns:
            True if all required libraries are installed
        """
        return True  # Override in subclasses if needed
    
    def get_required_libraries(self) -> List[str]:
        """
        Get list of required library names.
        
        Returns:
            List of library names (e.g., ['librosa', 'audioflux'])
        """
        return ['librosa']  # Default, override in subclasses
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SpectralFeatureExtractor(FeatureExtractor):
    """Base class for spectral analysis features."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
    
    def get_required_libraries(self) -> List[str]:
        return ['librosa', 'scipy']


class TemporalFeatureExtractor(FeatureExtractor):
    """Base class for temporal analysis features."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
    
    def get_required_libraries(self) -> List[str]:
        return ['librosa']


class HarmonicFeatureExtractor(FeatureExtractor):
    """Base class for harmonic analysis features."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
    
    def get_required_libraries(self) -> List[str]:
        return ['librosa']


class VocalFeatureExtractor(FeatureExtractor):
    """Base class for vocal forensics features."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
    
    def get_required_libraries(self) -> List[str]:
        return ['librosa', 'scipy']


# Utility functions for feature extractors

def load_audio(audio_path: str, sr: Optional[int] = None, 
               mono: bool = True) -> tuple:
    """
    Load audio file using librosa.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (None = native)
        mono: Convert to mono
    
    Returns:
        (y, sr) tuple
    """
    import librosa
    return librosa.load(audio_path, sr=sr, mono=mono)


def normalize_score(value: float, low_threshold: float, 
                    high_threshold: float, invert: bool = False) -> float:
    """
    Normalize a metric value to 0-1 score.
    
    Args:
        value: Raw metric value
        low_threshold: Value that maps to 0 (or 1 if inverted)
        high_threshold: Value that maps to 1 (or 0 if inverted)
        invert: If True, low values = high score
    
    Returns:
        Normalized score between 0 and 1
    """
    if low_threshold == high_threshold:
        return 0.5
    
    # Linear interpolation
    score = (value - low_threshold) / (high_threshold - low_threshold)
    score = np.clip(score, 0.0, 1.0)
    
    if invert:
        score = 1.0 - score
    
    return float(score)


def gaussian_score(value: float, peak: float, width: float, 
                   amplitude: float = 1.0) -> float:
    """
    Calculate Gaussian-shaped score around a peak value.
    
    Useful for detecting suspicious values (e.g., 16kHz cutoff).
    
    Args:
        value: Input value
        peak: Center of Gaussian (most suspicious value)
        width: Standard deviation
        amplitude: Maximum score at peak
    
    Returns:
        Score between 0 and amplitude
    """
    distance = abs(value - peak)
    score = amplitude * np.exp(-(distance ** 2) / (2 * width ** 2))
    return float(score)
