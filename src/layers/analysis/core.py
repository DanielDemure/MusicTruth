"""
Analyzer Module.

Orchestrates the feature extraction process based on selected Analysis Mode.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from src.utils.logger import logger

from src.config import config, AnalysisMode, Genre, GENRE_PROFILES
from .features.base import FeatureResult, load_audio

class Analyzer:
    """
    Main analysis engine.
    Orchestrates feature extraction and ML inference.
    """
    
    def __init__(self):
        self.extractors = {}
        # Delay loading extractors until needed or instantiated
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
            from .features.essentia_extractor import get_essentia_extractors
            from .features.transcription import get_midi_extractors as get_transcription_extractors
            from .features.deepfake import get_dl_detectors
            from .features.forensic import get_forensic_extractors
            
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
                get_dl_detectors,
                get_forensic_extractors
            ]
            
            for getter in all_getters:
                for extractor in getter():
                    if extractor.is_available():
                        self.extractors[extractor.name] = extractor
                        
        except ImportError as e:
            logger.warning(f"Could not load some extractors: {e}")

    def analyze_audio(self, file_path: str, mode: AnalysisMode = AnalysisMode.STANDARD, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze audio file using features defined by the mode and metadata for adaptation.
        
        Args:
            file_path (str): Path to the audio file.
            mode (AnalysisMode): The depth of analysis to perform.
            metadata (Optional[Dict]): Metadata including genre for adaptation.
            
        Returns:
            Dict[str, Any]: Analysis results including features and AI probability.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 1. Load Audio
        try:
            y, sr = load_audio(file_path, sr=22050)
        except Exception as e:
            logger.error(f"Audio load failed for {file_path}: {e}")
            return {"error": f"Audio load failed: {e}"}
            
        results = {
            "filename": os.path.basename(file_path),
            "mode": mode.value,
            "features": {},
            "flags": [],
            "ai_probability": 0.0,
            "metadata": metadata or {}
        }
        
        scores = []
        
        # 2. Run Extractors
        # Special handling for FORENSIC mode: Separate stems first
        input_file_map = {"mix": file_path}
        
        if mode == AnalysisMode.FORENSIC:
            try:
                from src.layers.processing.separation import separate_audio
                # Use htdemucs_ft as per forensic report recommendation
                logger.info(f"Running forensic stem separation for {os.path.basename(file_path)}...")
                stems = separate_audio(file_path, model_name="htdemucs_ft")
                if stems:
                    input_file_map.update(stems)
                    logger.info(f"Stems available: {list(stems.keys())}")
            except Exception as e:
                logger.error(f"Forensic separation failed: {e}")

        for name, extractor in self.extractors.items():
            should_run = False
            target_audio = input_file_map["mix"] # Default to mix
            
            # Forensic extractors run on specific stems if available, or mix if not
            if "forensic" in name:
                if mode == AnalysisMode.FORENSIC:
                    should_run = True
                    # Silence analysis targets 'other' (piano) or 'vocals' if available
                    # Entropy analysis targets 'other' (piano) or mix
                    if "silence" in name and "other" in input_file_map:
                        target_audio = input_file_map["other"]
                    elif "entropy" in name and "other" in input_file_map:
                        target_audio = input_file_map["other"]
            
            elif mode == AnalysisMode.FORENSIC:
                should_run = True
            elif mode == AnalysisMode.DEEP:
                should_run = True
            elif mode == AnalysisMode.STANDARD:
                if not any(k in name for k in ['structural', 'midi', 'provider', 'forensic']):
                    should_run = True
            elif mode == AnalysisMode.QUICK:
                if any(k in name for k in ['cutoff', 'peak', 'tempo']):
                    should_run = True
            elif mode == AnalysisMode.CUSTOM:
                should_run = True
                
            if should_run:
                try:
                    # Load target audio if it's different from the mix we already loaded
                    current_y, current_sr = y, sr
                    
                    if target_audio != file_path:
                        # We need to load the stem
                        # Check if we should cache this? For now, load on demand (stems are usually smaller)
                        try:
                            current_y, current_sr = load_audio(target_audio, sr=22050)
                        except Exception as e:
                            logger.error(f"Could not load stem {target_audio}: {e}")
                            continue

                    res = extractor.extract(target_audio, y=current_y, sr=current_sr)
                    results["features"][name] = res.to_dict()
                    
                    if res.flags:
                        results["flags"].extend(res.flags)
                        
                except Exception as e:
                    logger.error(f"Extractor {name} failed: {e}")
                    
        # 3. Calculate Aggregate Score with Genre Adaptation
        genre_str = results.get('metadata', {}).get('genre', 'general')
        try:
            genre_name = Genre(genre_str)
        except ValueError:
            genre_name = Genre.GENERAL
            
        profile = GENRE_PROFILES.get(genre_name, GENRE_PROFILES[Genre.GENERAL])
        logger.debug(f"Using adaptation profile for genre: {genre_name.value}")

        for name, res_dict in results["features"].items():
            if res_dict.get('score', 0) > 0:
                score = res_dict['score']
                weight = 1.0
                
                if "ml" in name or "provider" in name:
                    weight = 2.0
                
                # Report recommends high weight for Silence (30%) and Entropy (20%)
                if "forensic" in name:
                    weight = 4.0 
                
                if genre_name != Genre.GENERAL:
                    if "tempo" in name:
                        weight *= profile.tempo_stability_weight
                    elif "mfcc" in name:
                        weight *= profile.mfcc_uniformity_weight
                    elif "contrast" in name:
                        weight *= profile.spectral_contrast_weight
                    elif "vocal" in name or "quantization" in name:
                        weight *= profile.vocal_artifact_weight
                
                scores.extend([score] * int(max(1, weight * 10)))

        if scores:
            results["ai_probability"] = float(np.mean(scores))
        else:
            results["ai_probability"] = 0.0
            
        return results
