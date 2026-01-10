"""
Transcription Feature Extractor.

Uses Spotify's Basic Pitch to transcribe audio to MIDI, 
allowing for musicological analysis via music21.
"""

import os
import tempfile
import numpy as np
from typing import Optional, Dict, Any, List

from .base import FeatureExtractor, FeatureResult

try:
    from basic_pitch.inference import predict_and_save, predict
    import pretty_midi
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


class TranscriptionExtractor(FeatureExtractor):
    """
    Transcribes audio to MIDI and performs basic musicological analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="midi_extraction",
            description="Audio-to-MIDI transcription using Basic Pitch"
        )
        
    def is_available(self) -> bool:
        return BASIC_PITCH_AVAILABLE and MUSIC21_AVAILABLE
        
    def get_required_libraries(self) -> List[str]:
        return ['basic_pitch', 'music21', 'pretty_midi']
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        
        if not self.is_available():
            return FeatureResult(self.name, metrics={'error': 'Dependencies missing (basic-pitch, music21)'})
            
        try:
            # Predict MIDI data
            # basic_pitch predict returns: model_output, midi_data, note_events
            # we need the midi_data (pretty_midi object)
            
            # Note: Basic Pitch works best with file path
            if not audio_path or not os.path.exists(audio_path):
                 return FeatureResult(self.name, metrics={'error': 'Audio file required for Basic Pitch'})

            model_output, midi_data, note_events = predict(audio_path)
            
            # Analyze MIDI with music21
            # We can export pretty_midi to a temp file and load with music21
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                midi_data.write(tmp.name)
                tmp_path = tmp.name
                
            try:
                score = music21.converter.parse(tmp_path)
                
                # Basic Musicological Features
                # 1. Key Estimate (music21)
                key = score.analyze('key')
                
                # 2. Time Signature
                ts = score.getTimeSignatures()[0] if score.getTimeSignatures() else "Unknown"
                
                # 3. Note Density
                total_notes = len(score.flatten().notes)
                duration_secs = midi_data.get_end_time()
                notes_per_sec = total_notes / duration_secs if duration_secs > 0 else 0
                
                # 4. Melodic Interval Analysis (detect robotic steps?)
                # Simplified: just return basic stats
                
                metrics = {
                    'estimated_key': f"{key.tonic.name} {key.mode}",
                    'key_confidence': key.correlationCoefficient,
                    'time_signature': f"{ts.numerator}/{ts.denominator}" if hasattr(ts, 'numerator') else str(ts),
                    'total_notes': total_notes,
                    'notes_per_second': notes_per_sec
                }
                
                return FeatureResult(
                    feature_name=self.name,
                    score=0.0, # Neutral score, used for metadata mostly
                    metrics=metrics
                )
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            return FeatureResult(self.name, metrics={'error': f"Transcription failed: {e}"})

def get_midi_extractors():
    return [TranscriptionExtractor()]
