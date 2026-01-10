"""
Vocal feature extractors for AI music detection.

Analyzes isolated vocal tracks for signs of AI generation,
including perfect pitch quantization, lack of breaths, and
unnatural vibrato.
"""

import numpy as np
import librosa
import scipy.stats
from typing import Optional, Tuple

from .base import VocalFeatureExtractor, FeatureResult, load_audio, normalize_score


class PitchQuantizationAnalyzer(VocalFeatureExtractor):
    """
    Detects "perfect pitch" artifacts (Auto-Tune effect).
    
    AI vocals often align perfectly to the nearest semitone with
    minimal natural drift.
    """
    
    def __init__(self):
        super().__init__(
            name="vocal_pitch",
            description="Analyzes pitch quantization and deviation"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract pitch quantization features."""
        # Note: input expects isolated vocals
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=None)
            
        # Estimate pitch using PYIN (robust f0 estimation)
        # Range C2 (65Hz) to C6 (1046Hz) for vocals
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C6')
        )
        
        # Filter only voiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': 'No voiced segments detected'},
                flags=["No vocals detected in track"]
            )
            
        # Calculate deviation from nearest semitone
        midi_pitch = librosa.hz_to_midi(f0_voiced)
        nearest_note = np.round(midi_pitch)
        deviation = np.abs(midi_pitch - nearest_note)
        
        avg_deviation = np.mean(deviation)
        std_deviation = np.std(deviation)
        
        # AI (and heavy Auto-Tune) < 0.1 semitones
        # Natural singing > 0.15 semitones
        
        score = 0.0
        if avg_deviation < 0.05:
            score = 0.9 # Extremely robotic
        elif avg_deviation < 0.1:
            score = 0.6
        else:
            score = 0.0
            
        flags = []
        if score > 0.5:
            flags.append(f"Unnaturally perfect pitch (Deviation: {avg_deviation:.3f} semitones)")
            
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.8,
            metrics={
                'pitch_deviation_mean': float(avg_deviation),
                'pitch_deviation_std': float(std_deviation),
                'voiced_fraction': float(np.sum(voiced_flag) / len(f0))
            },
            flags=flags
        )


class VibratoAnalyzer(VocalFeatureExtractor):
    """
    Analyzes vocal vibrato characteristics.
    
    Natural vibrato has specific rate (5-7Hz) and extent.
    AI vibrato might be absent, too regular, or have wrong rate.
    """
    
    def __init__(self):
        super().__init__(
            name="vocal_vibrato",
            description="Analyzes vibrato rate and extent"
        )
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=None)
            
        # Only feasible if we have pitch curve
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=1046)
        
        if np.sum(voiced_flag) < 100:
             return FeatureResult(self.name, metrics={'error': 'Insufficient vocal data'})

        # Interpolate unvoiced parts for continuity (simplified)
        # Ideally we segment into notes and analyze sustained notes
        
        # Placeholder for complex vibrato analysis
        # Detailed logic: 
        # 1. Segment f0 into continuous notes
        # 2. Extract AC component of f0 in 4-8Hz band
        # 3. Measure amplitude and frequency of that component
        
        return FeatureResult(
            feature_name=self.name,
            score=0.0,
            confidence=0.3,
            metrics={'status': 'Not fully implemented'},
            flags=["Vibrato analysis simplified"]
        )


class BreathDetector(VocalFeatureExtractor):
    """
    Detects natural breaths between phrases.
    
    AI models often generate continuous singing without pauss
    or breaths.
    """
    
    def __init__(self):
        super().__init__(
            name="vocal_breath",
            description="Detects absence of natural breaths"
        )
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
            
        # Breaths are unvoiced, high-frequency, noise-like segments
        # typically 200-500ms long.
        
        # 1. Compute RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # 2. Compute Spectral Flatness (breaths are noise-like -> high flatness)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        
        # 3. Simple heuristic detector
        # Energy > silent_thresh AND Energy < vocal_thresh
        # Flatness > noise_thresh
        
        silent_thresh = 0.001
        vocal_thresh = 0.05
        flatness_thresh = 0.3
        
        candidates = (rms > silent_thresh) & (rms < vocal_thresh) & (flatness > flatness_thresh)
        
        # Count segments
        # Use simple run length encoding or diff
        
        num_breaths = 0 # Placeholder
        
        # AI Detection: Long phrases (> 15s) with no breaths?
        
        # This requires robust voice activity detection (VAD) first.
        # Implemented simplified check for "Continuous Activity"
        
        active_ratio = np.sum(rms > silent_thresh) / len(rms)
        
        score = 0.0
        if active_ratio > 0.95:
             # Almost continuous sound
             score = 0.6
             
        flags = []
        if score > 0.5:
            flags.append("Unnaturally continuous vocals (few pauses/breaths)")
            
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.5,
            metrics={
                'active_ratio': float(active_ratio),
                'breath_count_est': 0 # To be implemented
            },
            flags=flags
        )


def get_vocal_extractors():
    """Get all available vocal feature extractors."""
    return [
        PitchQuantizationAnalyzer(),
        VibratoAnalyzer(),
        BreathDetector()
    ]
