"""
Temporal feature extractors for AI music detection.

Implements temporal analysis methods including tempo, rhythm,
and onset detection.
"""

import numpy as np
import librosa
import scipy.stats
from typing import Optional

from .base import TemporalFeatureExtractor, FeatureResult, load_audio, normalize_score


class TempoStabilityAnalyzer(TemporalFeatureExtractor):
    """
    Analyzes beat consistency to detect robotic quantization.
    
    AI-generated music often has unnaturally perfect timing,
    while human performances have natural micro-variations.
    """
    
    def __init__(self):
        super().__init__(
            name="tempo_stability",
            description="Detects unnaturally perfect tempo consistency"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract tempo stability features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Detect beats
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Ensure tempo is scalar
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
        
        if len(beats) < 2:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': 'Insufficient beats detected'},
                flags=["Could not detect sufficient beats for analysis"]
            )
        
        # Calculate beat intervals
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        
        if len(intervals) == 0:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'tempo_bpm': tempo}
            )
        
        # Calculate coefficient of variation (CV)
        std_dev = np.std(intervals)
        avg_interval = np.mean(intervals)
        
        if avg_interval == 0:
            cv = 0.0
        else:
            cv = std_dev / avg_interval
        
        # Calculate score
        score = self._calculate_score(cv)
        
        # Generate flags
        flags = []
        if score > 0.5:
            flags.append(f"High tempo stability detected (CV: {cv:.4f})")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.7,
            metrics={
                'tempo_bpm': tempo,
                'beat_interval_mean': float(avg_interval),
                'beat_interval_std': float(std_dev),
                'coefficient_of_variation': float(cv),
                'num_beats': len(beats)
            },
            flags=flags
        )
    
    def _calculate_score(self, cv: float) -> float:
        """
        Calculate score based on coefficient of variation.
        
        CV < 0.01: Very robotic (0.9)
        CV > 0.05: Very human (0.0)
        """
        if cv < 0.01:
            return 0.9
        elif cv > 0.05:
            return 0.0
        else:
            # Linear interpolation
            return 0.9 * (1.0 - (cv - 0.01) / 0.04)


class OnsetDetectionAnalyzer(TemporalFeatureExtractor):
    """
    Analyzes onset patterns for AI artifacts.
    
    Onset detection reveals how notes/sounds start. AI music
    may have unusual onset patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="onset_detection",
            description="Analyzes onset patterns for AI artifacts"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract onset features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        if len(onset_times) < 2:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'num_onsets': len(onset_times)}
            )
        
        # Calculate onset intervals
        onset_intervals = np.diff(onset_times)
        
        # Calculate statistics
        interval_mean = np.mean(onset_intervals)
        interval_std = np.std(onset_intervals)
        interval_cv = interval_std / interval_mean if interval_mean > 0 else 0
        
        # Check for grid-like quantization
        # AI might place onsets on a very regular grid
        grid_score = self._check_grid_quantization(onset_intervals)
        
        # Check for unnatural regularity
        regularity_score = 0.0
        if interval_cv < 0.2:  # Very regular
            regularity_score = 0.5
        
        score = max(grid_score, regularity_score)
        
        flags = []
        if grid_score > 0.5:
            flags.append("Grid-like onset quantization detected")
        if regularity_score > 0.4:
            flags.append(f"Unusually regular onset spacing (CV: {interval_cv:.3f})")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.5,
            metrics={
                'num_onsets': len(onset_times),
                'onset_interval_mean': float(interval_mean),
                'onset_interval_std': float(interval_std),
                'onset_interval_cv': float(interval_cv),
                'grid_score': float(grid_score)
            },
            flags=flags
        )
    
    def _check_grid_quantization(self, intervals: np.ndarray) -> float:
        """Check if onsets fall on a regular grid."""
        # Calculate histogram of intervals
        # If many intervals are exact multiples of a base interval,
        # it suggests quantization
        
        if len(intervals) < 5:
            return 0.0
        
        # Find potential grid size (most common interval)
        hist, bin_edges = np.histogram(intervals, bins=20)
        most_common_interval = bin_edges[np.argmax(hist)]
        
        # Check how many intervals are close to multiples of this
        tolerance = most_common_interval * 0.1
        quantized_count = 0
        
        for interval in intervals:
            # Check if close to 1x, 2x, 3x, etc.
            for multiple in [1, 2, 3, 4]:
                expected = most_common_interval * multiple
                if abs(interval - expected) < tolerance:
                    quantized_count += 1
                    break
        
        quantized_ratio = quantized_count / len(intervals)
        
        # High ratio suggests grid quantization
        if quantized_ratio > 0.8:
            return 0.7
        elif quantized_ratio > 0.6:
            return 0.4
        return 0.0


class RhythmComplexityAnalyzer(TemporalFeatureExtractor):
    """
    Analyzes rhythm complexity.
    
    AI music may have simpler or more repetitive rhythmic patterns
    compared to human compositions.
    """
    
    def __init__(self):
        super().__init__(
            name="rhythm_complexity",
            description="Analyzes rhythmic complexity and repetition"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract rhythm complexity features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Get tempogram
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # Calculate complexity metrics
        # 1. Entropy of tempogram (higher = more complex)
        tempogram_flat = tempogram.flatten()
        tempogram_flat = tempogram_flat[tempogram_flat > 0]  # Remove zeros
        
        if len(tempogram_flat) == 0:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': 'No tempogram data'}
            )
        
        # Normalize to probability distribution
        tempogram_prob = tempogram_flat / np.sum(tempogram_flat)
        entropy = scipy.stats.entropy(tempogram_prob)
        
        # 2. Variance in onset strength
        onset_variance = np.var(onset_env)
        
        # Low entropy and low variance suggest simple/repetitive rhythm
        complexity_score = self._calculate_complexity_score(entropy, onset_variance)
        
        # Invert for AI score (low complexity = high AI score)
        ai_score = 1.0 - complexity_score
        
        flags = []
        if ai_score > 0.6:
            flags.append(f"Low rhythmic complexity (entropy: {entropy:.2f})")
        
        return FeatureResult(
            feature_name=self.name,
            score=ai_score,
            confidence=0.4,  # Lower confidence, rhythm is subjective
            metrics={
                'tempogram_entropy': float(entropy),
                'onset_variance': float(onset_variance),
                'complexity_score': float(complexity_score)
            },
            flags=flags
        )
    
    def _calculate_complexity_score(self, entropy: float, variance: float) -> float:
        """Calculate complexity score (0 = simple, 1 = complex)."""
        # Normalize entropy (typical range 0-10)
        entropy_norm = min(entropy / 10.0, 1.0)
        
        # Normalize variance (typical range 0-100)
        variance_norm = min(variance / 100.0, 1.0)
        
        # Combine (equal weight)
        return (entropy_norm + variance_norm) / 2.0


class BeatHistogramAnalyzer(TemporalFeatureExtractor):
    """
    Analyzes beat histogram for periodicity patterns.
    
    Examines the distribution of beat strengths over time.
    """
    
    def __init__(self):
        super().__init__(
            name="beat_histogram",
            description="Analyzes beat strength distribution"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract beat histogram features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Get onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Calculate statistics
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)
        onset_max = np.max(onset_env)
        
        # Check for unnatural uniformity
        cv = onset_std / onset_mean if onset_mean > 0 else 0
        
        uniformity_score = 0.0
        if cv < 0.3:  # Very uniform beat strengths
            uniformity_score = 0.6
        elif cv < 0.5:
            uniformity_score = 0.3
        
        flags = []
        if uniformity_score > 0.5:
            flags.append(f"Unusually uniform beat strengths (CV: {cv:.3f})")
        
        return FeatureResult(
            feature_name=self.name,
            score=uniformity_score,
            confidence=0.5,
            metrics={
                'onset_strength_mean': float(onset_mean),
                'onset_strength_std': float(onset_std),
                'onset_strength_max': float(onset_max),
                'onset_strength_cv': float(cv)
            },
            flags=flags
        )


# Factory function
def get_temporal_extractors():
    """Get all available temporal feature extractors."""
    return [
        TempoStabilityAnalyzer(),
        OnsetDetectionAnalyzer(),
        RhythmComplexityAnalyzer(),
        BeatHistogramAnalyzer()
    ]
