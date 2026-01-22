"""
Feature extractors for forensic analysis (Silence, Entropy, Pitch Patterns).
Based on the "Dusk Awakened by Dawn" Technical Analysis Report.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, List, Optional
from src.utils.logger import logger
from .base import FeatureExtractor, FeatureResult, load_audio, normalize_score

class SilenceExtractor(FeatureExtractor):
    """
    Analyzes silence patterns to distinguish human expressive pauses from AI uniformity.
    
    Metrics:
    - Silence Percentage
    - Mean Silence Duration
    - Expressive Gaps (pauses > 0.5s)
    """
    
    def __init__(self):
        super().__init__("silence_forensics", "Silence pattern analysis")
        
    def get_required_libraries(self) -> List[str]:
        return ['librosa', 'scipy']
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None, sr: Optional[int] = None, **kwargs) -> FeatureResult:
        import librosa
        if y is None:
            y, sr = load_audio(audio_path)
            
        # 1. Detect Silence
        # We use a threshold relative to the max amplitude or absolute dB
        # Report used -50dBFS. Librosa uses amplitude. -50dB ~= 0.003 amplitude
        threshold_db = 50
        top_db = threshold_db
        
        # Split into non-silent intervals
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
        
        # Calculate silence stats
        total_samples = len(y)
        non_silent_samples = sum(end - start for start, end in intervals)
        silent_samples = total_samples - non_silent_samples
        
        silence_pct = (silent_samples / total_samples) * 100
        
        # Calculate duration of GAPS (silences between intervals)
        gap_durations = []
        if len(intervals) > 1:
            for i in range(len(intervals) - 1):
                # Gap is between end of current and start of next
                gap_len = intervals[i+1][0] - intervals[i][1]
                gap_durations.append(gap_len / sr)
        
        mean_gap = np.mean(gap_durations) if gap_durations else 0.0
        std_gap = np.std(gap_durations) if gap_durations else 0.0
        expressive_gaps = sum(1 for g in gap_durations if g > 0.5)
        
        # Scoring Logic based on Report
        # AI: 11-18% silence, short gaps < 0.11s
        # Human: 14-90% silence, long gaps
        
        # We flag "Likely AI" if silence is LOW AND gaps are SHORT
        suspicion = 0.0
        
        if silence_pct < 20: 
            suspicion += 0.4
        elif silence_pct > 80: # Very sparse can be human
            suspicion -= 0.2
            
        if mean_gap < 0.15:
            suspicion += 0.4
            
        suspicion = min(max(suspicion, 0.0), 1.0)
        
        metrics = {
            "silence_percentage": float(silence_pct),
            "mean_gap_duration": float(mean_gap),
            "std_gap_duration": float(std_gap),
            "expressive_gaps_count": int(expressive_gaps),
            "gap_count": len(gap_durations)
        }
        
        flags = []
        if suspicion > 0.5:
            flags.append(f"Low silence/short gaps detected ({silence_pct:.1f}%, {mean_gap:.3f}s)")
            
        return FeatureResult(
            feature_name=self.name,
            score=suspicion,
            metrics=metrics,
            flags=flags
        )

class EntropyExtractor(FeatureExtractor):
    """
    Analyzes Shannon entropy of musical information to detect encoding or algorithmic generation.
    """
    
    def __init__(self):
        super().__init__("entropy_forensics", "Shannon entropy of pitch classes")
        
    def get_required_libraries(self) -> List[str]:
        return ['librosa', 'scipy']
        
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None, sr: Optional[int] = None, **kwargs) -> FeatureResult:
        import librosa
        if y is None:
            y, sr = load_audio(audio_path)
            
        # Chroma Features (Pitch Classes)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Flatten and normalize to get probability distribution
        # We aggregate over time to look for global distribution anomalies
        # OR we look at the sequence as a stream (as per report for "encoding")
        
        # Report method: "Mapped notes to binary... Calculated Shannon entropy"
        # We'll use a simplified version: Entropy of the Chroma Energy Distribution
        
        chroma_flat = chroma.flatten()
        chroma_sum = np.sum(chroma_flat)
        if chroma_sum == 0:
            return FeatureResult(self.name, 0.0, metrics={"entropy": 0.0})
            
        prob_dist = chroma_flat / chroma_sum
        prob_dist = prob_dist[prob_dist > 0] # Avoid log(0)
        
        entropy = -np.sum(prob_dist * np.log2(prob_dist))
        
        # Normalizing entropy?
        # Max entropy for 12 bins? No, this is flattened over time. 
        # A perfectly random signal has very high entropy.
        # A highly structured (single note repeating) has low entropy.
        
        # Report: 
        # AI: ~0.50
        # Letter (Encoded): 0.09 (Super low)
        # Human Complex: > 0.88
        
        # Note: Their entropy scale might be different based on binary mapping.
        # We will use relative deviation from expected "musical" entropy.
        
        metrics = {
            "shannon_entropy": float(entropy),
            "distribution_size": len(prob_dist)
        }
        
        suspicion = 0.0
        flags = []
        
        # Heuristics based on general observation
        # Extremely low entropy = simple loops or encoding (Suspicious)
        # Mid-range (0.4-0.6 on normalized scale) could be AI generic
        # High = Complex audio
        
        # We simplify: standard music usually has variety.
        if entropy < 1.0: # Very low information content
             suspicion = 0.8
             flags.append(f"Extremely low entropy ({entropy:.2f}): possible encoding or repetitive loop")
        
        return FeatureResult(
            feature_name=self.name,
            score=suspicion,
            metrics=metrics,
            flags=flags
        )

def get_forensic_extractors() -> List[FeatureExtractor]:
    return [SilenceExtractor(), EntropyExtractor()]
