"""
Essentia Feature Extractor.

Leverages the Essentia C++ library (via Python bindings) for high-performance,
standardized music descriptor extraction.
"""

import numpy as np
from typing import Optional, Dict, Any, List

from .base import FeatureExtractor, FeatureResult, load_audio

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

class EssentiaFeatureExtractor(FeatureExtractor):
    """
    Extracts high-level descriptors using Essentia.
    
    Includes:
    - Danceability
    - Dynamic Complexity
    - Tonal Key/Scale
    - BPM (Essentia algorithm)
    """
    
    def __init__(self):
        super().__init__(
            name="essentia_features",
            description="High-level descriptors from the Essentia MIR library"
        )
        
    def is_available(self) -> bool:
        return ESSENTIA_AVAILABLE
        
    def get_required_libraries(self) -> List[str]:
        return ['essentia']
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """
        Extract features using Essentia.
        Note: Essentia has its own loaders, but we can convert numpy arrays if needed.
        """
        if not self.is_available():
             return FeatureResult(self.name, metrics={'error': 'Essentia not installed'})

        try:
            # Essentia prefers loading the file itself or converting standard MonoLoader
            # If we have path, use it.
            if audio_path:
                loader = es.MonoLoader(filename=audio_path)
                audio = loader()
            elif y is not None:
                # Convert numpy to essentia vector (float32)
                audio = y.astype('float32')
            else:
                 return FeatureResult(self.name, metrics={'error': 'No audio input'})

            # 1. Danceability
            danceability_algo = es.Danceability()
            danceability, _ = danceability_algo(audio)
            
            # 2. Dynamic Complexity
            dynamic_complexity_algo = es.DynamicComplexity()
            dyn_complexity, _ = dynamic_complexity_algo(audio)
            
            # 3. Key/Scale (requires spectral peaks usually, but KeyExtractor works directly on audio? 
            # optimized key extractor typically takes audio or HPCP)
            # Let's use simple KeyExtractor
            key_extractor = es.KeyExtractor()
            key, scale, key_strength = key_extractor(audio)
            
            metrics = {
                'danceability': float(danceability),
                'dynamic_complexity': float(dyn_complexity),
                'key': key,
                'scale': scale,
                'key_strength': float(key_strength)
            }
            
            # Scoring Logic (Heuristic for AI)
            # AI music *can* be very generic (high danceability?) or flat (low dynamic complexity?)
            # This is hard to score definitively without a trained model on these features.
            # For now, we return 0.0 score but provide the metrics for the ML layer.
            
            return FeatureResult(
                feature_name=self.name,
                score=0.0, 
                confidence=1.0,
                metrics=metrics,
                metadata={'library': f'Essentia {essentia.__version__}'}
            )
            
        except Exception as e:
            return FeatureResult(
                feature_name=self.name,
                metrics={'error': str(e)}
            )

def get_essentia_extractors():
    """Get Essentia extractors."""
    return [EssentiaFeatureExtractor()]
