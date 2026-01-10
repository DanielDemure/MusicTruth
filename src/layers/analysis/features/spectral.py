"""
Spectral feature extractors for AI music detection.

Implements various spectral analysis methods to detect artifacts
common in AI-generated music.
"""

import numpy as np
import librosa
import scipy.stats
from typing import Optional, Dict, Any

from .base import SpectralFeatureExtractor, FeatureResult, load_audio, gaussian_score, normalize_score

# Try importing audioFlux for advanced features
try:
    import audioflux as af
    AUDIOFLUX_AVAILABLE = True
except ImportError:
    AUDIOFLUX_AVAILABLE = False


class FrequencyCutoffDetector(SpectralFeatureExtractor):
    """
    Detects hard frequency cutoffs common in AI upsampling.
    
    AI models often have training limitations that result in sharp
    cutoffs at specific frequencies (e.g., 16kHz, 20kHz).
    """
    
    def __init__(self):
        super().__init__(
            name="frequency_cutoff",
            description="Detects hard frequency cutoffs from AI upsampling"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract frequency cutoff features."""
        # Load audio if not provided
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=None)  # Native SR
        
        # Compute spectral rolloff at 99th percentile
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        cutoff_freq = np.mean(rolloff)
        
        # Calculate score based on suspicious frequencies
        score = self._calculate_score(cutoff_freq)
        
        # Generate flags
        flags = []
        if score > 0.5:
            flags.append(f"Hard frequency cutoff detected at {cutoff_freq/1000:.1f}kHz")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.8,  # High confidence for this method
            metrics={
                'cutoff_frequency_hz': float(cutoff_freq),
                'cutoff_frequency_khz': float(cutoff_freq / 1000),
                'rolloff_std': float(np.std(rolloff))
            },
            flags=flags
        )
    
    def _calculate_score(self, cutoff_freq: float) -> float:
        """
        Calculate AI suspicion score based on cutoff frequency.
        
        Suspicious frequencies:
        - < 10kHz: Very poor quality (0.9)
        - ~16kHz: Common AI/MP3 artifact (0.8)
        - ~20kHz: Suspicious (0.4)
        - > 21kHz: Normal (0.0)
        """
        if cutoff_freq < 10000:
            return 0.9
        elif cutoff_freq > 21000:
            return 0.0
        else:
            # Gaussian peaks at suspicious frequencies
            score_16k = gaussian_score(cutoff_freq, peak=16000, width=1000, amplitude=0.8)
            score_20k = gaussian_score(cutoff_freq, peak=20000, width=1000, amplitude=0.4)
            return max(score_16k, score_20k)


class SpectralPeakDetector(SpectralFeatureExtractor):
    """
    Detects systematic spectral peaks (grid artifacts).
    
    Deconvolution layers in GANs/Diffusion models can create
    regular "comb filter" patterns in the high-frequency spectrum.
    """
    
    def __init__(self):
        super().__init__(
            name="spectral_peaks",
            description="Detects grid artifacts from neural network deconvolution"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract spectral peak features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=None)
        
        # Compute STFT
        D = np.abs(librosa.stft(y))
        mean_spectrum = np.mean(D, axis=1)
        
        # Focus on high frequencies (> 10kHz) where artifacts are visible
        freqs = librosa.fft_frequencies(sr=sr)
        mask = freqs > 10000
        
        if not np.any(mask):
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.0,
                metrics={'error': 'No high frequencies available'}
            )
        
        high_freq_spectrum = mean_spectrum[mask]
        
        # Normalize
        if np.max(high_freq_spectrum) == 0:
            return FeatureResult(
                feature_name=self.name,
                score=0.0,
                confidence=0.5,
                metrics={'high_freq_energy': 0.0}
            )
        
        norm_spec = high_freq_spectrum / np.max(high_freq_spectrum)
        
        # Calculate "spikiness" using second derivative
        d2 = np.diff(norm_spec, 2)
        peak_variance = np.var(d2)
        
        # Calculate score
        score = self._calculate_score(peak_variance)
        
        # Generate flags
        flags = []
        if score > 0.5:
            flags.append(f"Systematic spectral peaks detected (Var: {peak_variance:.2e})")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.7,
            metrics={
                'peak_variance': float(peak_variance),
                'high_freq_mean': float(np.mean(high_freq_spectrum)),
                'high_freq_std': float(np.std(high_freq_spectrum))
            },
            flags=flags
        )
    
    def _calculate_score(self, peak_variance: float) -> float:
        """Calculate score based on peak variance."""
        if peak_variance > 1e-4:
            return 0.9
        elif peak_variance > 1e-5:
            return 0.5
        else:
            return 0.0


class MFCCAnalyzer(SpectralFeatureExtractor):
    """
    Analyzes Mel-Frequency Cepstral Coefficients for anomalies.
    
    AI-generated music may have unusual MFCC patterns compared
    to natural recordings.
    """
    
    def __init__(self):
        super().__init__(
            name="mfcc_analysis",
            description="Analyzes MFCC patterns for AI artifacts"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract MFCC features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)  # Standard SR for MFCCs
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = librosa.feature.delta(mfccs)
        
        # Check for anomalies
        # AI music might have unusually low variance in MFCCs
        variance_score = self._check_variance_anomaly(mfcc_std)
        
        # Check for unnatural smoothness in deltas
        delta_variance = np.var(mfcc_delta)
        smoothness_score = self._check_smoothness(delta_variance)
        
        score = max(variance_score, smoothness_score)
        
        flags = []
        if variance_score > 0.5:
            flags.append("Unusually low MFCC variance detected")
        if smoothness_score > 0.5:
            flags.append("Unnaturally smooth MFCC transitions")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.6,  # Medium confidence
            metrics={
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'mfcc_delta_variance': float(delta_variance),
                'variance_score': float(variance_score),
                'smoothness_score': float(smoothness_score)
            },
            flags=flags
        )
    
    def _check_variance_anomaly(self, mfcc_std: np.ndarray) -> float:
        """Check if MFCC variance is suspiciously low."""
        avg_std = np.mean(mfcc_std)
        # Natural music typically has avg std > 10
        # AI might have lower variance
        if avg_std < 5:
            return 0.8
        elif avg_std < 8:
            return 0.4
        return 0.0
    
    def _check_smoothness(self, delta_variance: float) -> float:
        """Check if MFCC deltas are too smooth."""
        # Very low delta variance indicates unnatural smoothness
        if delta_variance < 50:
            return 0.7
        elif delta_variance < 100:
            return 0.3
        return 0.0


class SpectralContrastAnalyzer(SpectralFeatureExtractor):
    """
    Analyzes spectral contrast for AI artifacts.
    
    Spectral contrast measures the difference between peaks and valleys
    in the spectrum. AI music may have unusual patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="spectral_contrast",
            description="Analyzes spectral contrast patterns"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract spectral contrast features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Compute spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Calculate statistics
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Check for anomalies
        # Unusually uniform contrast across frequency bands is suspicious
        uniformity_score = self._check_uniformity(contrast_std)
        
        # Check for extreme values
        extreme_score = self._check_extremes(contrast_mean)
        
        score = max(uniformity_score, extreme_score)
        
        flags = []
        if uniformity_score > 0.5:
            flags.append("Unusually uniform spectral contrast across bands")
        if extreme_score > 0.5:
            flags.append("Extreme spectral contrast values detected")
        
        return FeatureResult(
            feature_name=self.name,
            score=score,
            confidence=0.5,
            metrics={
                'contrast_mean': contrast_mean.tolist(),
                'contrast_std': contrast_std.tolist(),
                'uniformity_score': float(uniformity_score),
                'extreme_score': float(extreme_score)
            },
            flags=flags
        )
    
    def _check_uniformity(self, contrast_std: np.ndarray) -> float:
        """Check if contrast is too uniform across bands."""
        avg_std = np.mean(contrast_std)
        # Very low std indicates unnatural uniformity
        if avg_std < 2:
            return 0.7
        elif avg_std < 4:
            return 0.3
        return 0.0
    
    def _check_extremes(self, contrast_mean: np.ndarray) -> float:
        """Check for extreme contrast values."""
        max_contrast = np.max(contrast_mean)
        min_contrast = np.min(contrast_mean)
        
        # Very high or very low values are suspicious
        if max_contrast > 40 or min_contrast < -10:
            return 0.6
        return 0.0


class ZeroCrossingRateAnalyzer(SpectralFeatureExtractor):
    """
    Analyzes zero-crossing rate patterns.
    
    ZCR is related to the noisiness and percussiveness of audio.
    AI music may have unusual ZCR patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="zcr_analysis",
            description="Analyzes zero-crossing rate patterns"
        )
    
    def extract(self, audio_path: str, y: Optional[np.ndarray] = None,
                sr: Optional[int] = None, **kwargs) -> FeatureResult:
        """Extract ZCR features."""
        if y is None or sr is None:
            y, sr = load_audio(audio_path, sr=22050)
        
        # Compute zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate statistics
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        zcr_range = np.max(zcr) - np.min(zcr)
        
        # Check for anomalies
        # Very low variance in ZCR is suspicious
        variance_score = 0.0
        if zcr_std < 0.01:
            variance_score = 0.6
        elif zcr_std < 0.02:
            variance_score = 0.3
        
        flags = []
        if variance_score > 0.5:
            flags.append(f"Unusually stable ZCR (std: {zcr_std:.4f})")
        
        return FeatureResult(
            feature_name=self.name,
            score=variance_score,
            confidence=0.4,  # Lower confidence, ZCR alone is weak indicator
            metrics={
                'zcr_mean': float(zcr_mean),
                'zcr_std': float(zcr_std),
                'zcr_range': float(zcr_range)
            },
            flags=flags
        )


# Factory function to get all spectral extractors
def get_spectral_extractors():
    """Get all available spectral feature extractors."""
    return [
        FrequencyCutoffDetector(),
        SpectralPeakDetector(),
        MFCCAnalyzer(),
        SpectralContrastAnalyzer(),
        ZeroCrossingRateAnalyzer()
    ]
