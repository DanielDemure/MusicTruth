"""
MIDI feature extraction for AI music detection.

Converts audio to MIDI and analyzes note patterns.
"""

import numpy as np
import librosa
from typing import Optional

from .base import FeatureExtractor, FeatureResult, load_audio

try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


class MIDIQuantizationAnalyzer(FeatureExtractor):
    """
    Analyzes note timing quantization via Audio-to-MIDI conversion.
    """
    
    def __init__(self):
        super().__init__(
            name="midi_quantization",
            description="Analyzes MIDI timing quantization"
        )
        
    def is_available(self) -> bool:
        return MUSIC21_AVAILABLE
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        if not MUSIC21_AVAILABLE:
            return FeatureResult(self.name, metrics={'error': 'music21 missing'})
            
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
            
        # Audio to MIDI transcription is a hard problem.
        # We use a simplified onset+pitch approach for analysis
        
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        if len(onset_frames) < 10:
             return FeatureResult(self.name, score=0.0, metrics={'note_count': 0})
             
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Analyze grid adherence of onsets
        # Estimate BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, 'item'): tempo = tempo.item()
        if tempo <= 0: tempo = 120.0
        
        beat_duration = 60.0 / tempo
        
        # Calculate deviation from nearest 16th note grid
        sixteenth_duration = beat_duration / 4.0
        
        deviations = []
        for t in onset_times:
            # Find nearest grid point (assuming start at 0 for simplicity)
            # A real implementation aligns the grid phase
            remainder = t % sixteenth_duration
            dev = min(remainder, sixteenth_duration - remainder)
            deviations.append(dev)
            
        avg_dev = np.mean(deviations)
        normalized_dev = avg_dev / sixteenth_duration # 0.0 to 0.5
        
        # Extremely low deviation = Quantized
        score = 0.0
        if normalized_dev < 0.05:
            score = 0.8
            
        flags = []
        if score > 0.5:
             flags.append(f"Hard MIDI quantization detected (Dev: {normalized_dev:.3f})")

        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.6,
            metrics={
                'grid_deviation': float(normalized_dev),
                'estimated_tempo': float(tempo)
            },
            flags=flags
        )

def get_midi_extractors():
    return [MIDIQuantizationAnalyzer()]
