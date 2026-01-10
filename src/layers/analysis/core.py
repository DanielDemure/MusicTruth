"""
Analyzer Module.

Orchestrates the feature extraction process based on selected Analysis Mode.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional

from ...config import config, AnalysisMode
from .features.base import FeatureResult, load_audio

# Import Extractor Getters
# We verify these imports exist, otherwise we handle gracefully
try:
    from .features.spectral import get_spectral_extractors
    from .features.temporal import get_temporal_extractors
    from .features.harmonic import get_harmonic_extractors
    from .features.vocal import get_vocal_extractors
    from .features.structural import get_structural_extractors
    from .features.midi_features import get_midi_extractors
    from .features.provider_fingerprint import get_provider_extractors
except ImportError:
    # Fallback for when we run tests from different context
    pass

class Analyzer:
    """
    Main analysis engine.
    Orchestrates feature extraction and ML inference.
    """
    
    def __init__(self):
        self.extractors = {}
        # Delay loading extractors until needed or instantiated
        # to avoid circular imports during startup if dependencies missing
        self._load_extractors()
        
    def _load_extractors(self):
        """Load all available feature extractors into a map."""
        try:
            from .features.spectral import get_spectral_extractors
            from .features.temporal import get_temporal_extractors
            from .features.harmonic import get_harmonic_extractors
            from .features.vocal import get_vocal_extractors
            from .features.structural import get_structural_extractors
            from .features.midi_features import get_midi_extractors
            from .features.provider_fingerprint import get_provider_extractors
            # New Upgrades
            from .features.essentia_extractor import get_essentia_extractors
            from .features.transcription import get_midi_extractors as get_transcription_extractors
            from .features.deepfake import get_dl_detectors
            
            all_getters = [
                get_spectral_extractors,
                get_temporal_extractors,
                get_harmonic_extractors,
                get_vocal_extractors,
                get_structural_extractors,
                get_midi_extractors,
                get_provider_extractors,
                get_essentia_extractors,
                get_transcription_extractors,
                get_dl_detectors
            ]
            
            for getter in all_getters:
                for extractor in getter():
                    if extractor.is_available():
                        self.extractors[extractor.name] = extractor
                        
        except ImportError as e:
            print(f"Warning: Could not load some extractors: {e}")

    def analyze_audio(self, file_path: str, mode: AnalysisMode = AnalysisMode.STANDARD) -> Dict[str, Any]:
        """
        Analyze audio file using features defined by the mode.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 1. Load Audio
        try:
            # We load standard 22k mono for most checks.
            # Extractors needing high-res will reload or we can optimize this later
            y, sr = load_audio(file_path, sr=22050)
        except Exception as e:
            return {"error": f"Audio load failed: {e}"}
            
        results = {
            "filename": os.path.basename(file_path),
            "mode": mode.value,
            "features": {},
            "flags": [],
            "ai_probability": 0.0
        }
        
        scores = []
        
        # 2. Run Extractors
        # We run extractors that seem relevant to the mode.
        # Since we haven't mapped name-to-mode perfectly yet, we use heuristics:
        
        for name, extractor in self.extractors.items():
            should_run = False
            
            if mode == AnalysisMode.FORENSIC:
                should_run = True
            elif mode == AnalysisMode.DEEP:
                 # Deep runs most things, maybe skips very slow provider fingerprinting if not requested?
                 should_run = True
            elif mode == AnalysisMode.STANDARD:
                # Skip heavy structural/midi/provider
                if not any(k in name for k in ['structural', 'midi', 'provider']):
                    should_run = True
            elif mode == AnalysisMode.QUICK:
                # Only fastest
                if any(k in name for k in ['cutoff', 'peak', 'tempo']):
                    should_run = True
            elif mode == AnalysisMode.CUSTOM:
                should_run = True # Default for now
                
            if should_run:
                try:
                    # Pass the pre-loaded audio
                    # Note: some extractors might ignore y/sr if they need different settings
                    res = extractor.extract(file_path, y=y, sr=sr)
                    results["features"][name] = res.to_dict()
                    
                    if res.flags:
                        results["flags"].extend(res.flags)
                    
                    # Weighting logic (simplified)
                    if res.score > 0:
                        # Weight ML higher
                        weight = 2.0 if "ml" in name or "provider" in name else 1.0
                        scores.extend([res.score] * int(weight))
                        
                except Exception as e:
                    print(f"Extractor {name} failed: {e}")
                    
        # 3. Calculate Aggregate Score
        if scores:
            results["ai_probability"] = float(np.mean(scores))
        else:
            results["ai_probability"] = 0.0
            
        return results
