"""
MusicTruth Configuration Module

Centralized configuration for analysis modes, feature toggles, 
thresholds, and model settings.
"""

import os
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from src.utils.logger import logger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars only

# ============================================================================
# Analysis Modes
# ============================================================================

class AnalysisMode(Enum):
    """Analysis mode definitions with associated features and time estimates."""
    QUICK = "quick"           # 30s - Basic checks
    STANDARD = "standard"     # 2-5min - Comprehensive analysis
    DEEP = "deep"             # 10-30min - With source separation
    FORENSIC = "forensic"     # 30min-2hr - Full forensic suite
    CUSTOM = "custom"         # User-selected features


# ============================================================================
# Feature Flags
# ============================================================================

@dataclass
class FeatureFlags:
    """Feature availability flags based on installed libraries."""
    
    # Core libraries (always required)
    librosa_available: bool = True
    numpy_available: bool = True
    
    # Optional performance libraries
    audioflux_available: bool = False
    essentia_available: bool = False
    
    # ML libraries
    torch_available: bool = False
    transformers_available: bool = False
    
    # MIDI libraries
    music21_available: bool = False
    pretty_midi_available: bool = False
    
    # Visualization libraries
    plotly_available: bool = False
    seaborn_available: bool = False
    
    # Source separation
    demucs_available: bool = False
    
    def __post_init__(self):
        """Auto-detect available libraries."""
        self._detect_libraries()
    
    def _detect_libraries(self):
        """Detect which optional libraries are installed."""
        try:
            import audioflux
            self.audioflux_available = True
        except ImportError:
            logger.debug("AudioFlux not installed (optional).")
        
        try:
            import essentia
            self.essentia_available = True
        except ImportError:
            logger.debug("Essentia not installed (optional).")
        
        try:
            import torch
            self.torch_available = True
        except ImportError:
            logger.debug("PyTorch not installed. Deep learning features disabled.")
        
        try:
            import transformers
            self.transformers_available = True
        except ImportError:
            logger.debug("Transformers not installed. LLM/Deepfake features disabled.")
        
        try:
            import music21
            self.music21_available = True
        except ImportError:
            logger.debug("music21 not installed. MIDI analysis disabled.")
        
        try:
            import pretty_midi
            self.pretty_midi_available = True
        except ImportError:
            logger.debug("pretty_midi not installed. MIDI analysis disabled.")
        
        try:
            import plotly
            self.plotly_available = True
        except ImportError:
            logger.debug("Plotly not installed (optional).")
        
        try:
            import seaborn
            self.seaborn_available = True
        except ImportError:
            logger.debug("Seaborn not installed (optional).")
        
        try:
            import demucs
            self.demucs_available = True
        except ImportError:
            logger.debug("Demucs not installed. Source separation disabled.")


# ============================================================================
# Analysis Mode Configurations
# ============================================================================

MODE_FEATURES = {
    AnalysisMode.QUICK: [
        'spectral_cutoff',
        'tempo_stability',
        'ml_quick_check'
    ],
    
    AnalysisMode.STANDARD: [
        'spectral_cutoff',
        'spectral_peaks',
        'tempo_stability',
        'stereo_imaging',
        'silence_anomalies',
        'ml_full_inference'
    ],
    
    AnalysisMode.DEEP: [
        'spectral_cutoff',
        'spectral_peaks',
        'spectral_contrast',
        'mfcc_analysis',
        'chroma_analysis',
        'tempo_stability',
        'onset_detection',
        'stereo_imaging',
        'silence_anomalies',
        'harmonic_analysis',
        'key_detection',
        'chord_analysis',
        'source_separation',
        'vocal_forensics',
        'structural_analysis',
        'ml_full_inference'
    ],
    
    AnalysisMode.FORENSIC: [
        # All features from DEEP mode plus:
        'spectral_cutoff',
        'spectral_peaks',
        'spectral_contrast',
        'spectral_flatness',
        'mfcc_analysis',
        'chroma_analysis',
        'zcr_analysis',
        'tempo_stability',
        'onset_detection',
        'beat_histogram',
        'rhythm_complexity',
        'stereo_imaging',
        'silence_anomalies',
        'harmonic_analysis',
        'key_detection',
        'chord_analysis',
        'hpss_analysis',
        'tonal_centroid',
        'dissonance_analysis',
        'source_separation',
        'vocal_forensics',
        'vocal_breath',
        'vocal_vibrato',
        'vocal_formants',
        'structural_analysis',
        'segmentation',
        'novelty_analysis',
        'self_similarity',
        'midi_extraction',
        'midi_quantization',
        'provider_fingerprint_suno',
        'provider_fingerprint_udio',
        'provider_fingerprint_musicgen',
        'ml_ensemble',
        'segment_transformer'
    ]
}


# ============================================================================
# Detection Thresholds
# ============================================================================

@dataclass
class DetectionThresholds:
    """Threshold values for various detection methods."""
    
    # Spectral Analysis
    frequency_cutoff_suspicious: float = 16000  # Hz
    frequency_cutoff_very_low: float = 10000    # Hz
    spectral_peak_variance_high: float = 1e-4
    spectral_peak_variance_medium: float = 1e-5
    
    # Tempo Analysis
    tempo_cv_robotic: float = 0.01   # Very stable (robotic)
    tempo_cv_human: float = 0.05     # Natural variation
    
    # Stereo Analysis
    stereo_ratio_mono: float = 0.01  # Almost mono
    stereo_ratio_phase_issue: float = 1.0  # Phase problems
    
    # Vocal Analysis
    pitch_deviation_perfect: float = 0.05  # Unnaturally perfect
    pitch_deviation_natural: float = 0.15  # Natural singing
    
    # ML Confidence
    ml_confidence_high: float = 0.7
    ml_confidence_medium: float = 0.5
    ml_confidence_low: float = 0.3
    
    # Overall AI Probability
    ai_probability_high: float = 0.7   # Likely AI
    ai_probability_medium: float = 0.4  # Uncertain
    ai_probability_low: float = 0.2    # Likely human


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # Model cache directory
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"), ".cache", "musictruth", "models"
    ))
    
    # Deepfake detector model
    deepfake_model_id: str = "MelodyMachine/Deepfake-audio-detection-V2"
    
    # Segment Transformer (placeholder - to be implemented)
    segment_transformer_model_id: str = "segment-transformer-music"  # TBD
    
    # Model inference settings
    max_audio_length_seconds: int = 30  # For quick inference
    sample_rate: int = 16000  # Standard for most models
    
    # Ensemble settings
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'deepfake_detector': 0.4,
        'segment_transformer': 0.6
    })


# ============================================================================
# Path Configuration
# ============================================================================

@dataclass
class PathConfig:
    """File path configuration."""
    
    # Project root (auto-detected)
    project_root: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))
    
    @property
    def input_dir(self) -> str:
        return os.path.join(self.project_root, "input")
    
    @property
    def output_dir(self) -> str:
        return os.path.join(self.project_root, "output")
    
    @property
    def templates_dir(self) -> str:
        return os.path.join(self.project_root, "templates")
    
    @property
    def cache_dir(self) -> str:
        return os.path.join(self.project_root, ".cache")


# ============================================================================
# API Configuration
# ============================================================================

@dataclass
class APIConfig:
    """API keys and configuration loaded from environment variables."""
    
    # LLM Providers
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv('GEMINI_API_KEY'))
    gemini_model: str = field(default_factory=lambda: os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'))
    
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    openai_model: str = field(default_factory=lambda: os.getenv('OPENAI_MODEL', 'gpt-4'))
    
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY'))
    anthropic_model: str = field(default_factory=lambda: os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'))
    
    default_llm_provider: str = field(default_factory=lambda: os.getenv('DEFAULT_LLM_PROVIDER', 'gemini'))
    
    # Metadata APIs
    spotify_client_id: Optional[str] = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_ID'))
    spotify_client_secret: Optional[str] = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_SECRET'))
    musicbrainz_contact: Optional[str] = field(default_factory=lambda: os.getenv('MUSICBRAINZ_CONTACT_EMAIL'))
    
    # Analysis defaults
    default_analysis_mode: str = field(default_factory=lambda: os.getenv('DEFAULT_ANALYSIS_MODE', 'standard'))
    default_output_formats: str = field(default_factory=lambda: os.getenv('DEFAULT_OUTPUT_FORMATS', 'json,html'))
    
    def get_llm_config(self, provider: Optional[str] = None) -> tuple[Optional[str], str]:
        """Get API key and model for specified provider (or default)."""
        provider = provider or self.default_llm_provider
        
        if provider == 'gemini':
            return self.gemini_api_key, self.gemini_model
        elif provider == 'openai':
            return self.openai_api_key, self.openai_model
        elif provider == 'anthropic':
            return self.anthropic_api_key, self.anthropic_model
        else:
            return None, ''
    
    def has_spotify_credentials(self) -> bool:
        """Check if Spotify credentials are configured."""
        return bool(self.spotify_client_id and self.spotify_client_secret)


# ============================================================================
# Global Configuration Instance
# ============================================================================

class Config:
    """Global configuration singleton."""
    
    def __init__(self):
        self.features = FeatureFlags()
        self.thresholds = DetectionThresholds()
        self.models = ModelConfig()
        self.paths = PathConfig()
        self.api = APIConfig()
        self.mode_features = MODE_FEATURES
    
    def get_features_for_mode(self, mode: AnalysisMode) -> List[str]:
        """Get list of features for a given analysis mode."""
        return self.mode_features.get(mode, [])
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature can be used based on library availability."""
        # Map features to required libraries
        feature_requirements = {
            'audioflux_features': self.features.audioflux_available,
            'essentia_features': self.features.essentia_available,
            'ml_quick_check': self.features.torch_available and self.features.transformers_available,
            'ml_full_inference': self.features.torch_available and self.features.transformers_available,
            'ml_ensemble': self.features.torch_available and self.features.transformers_available,
            'segment_transformer': self.features.torch_available and self.features.transformers_available,
            'source_separation': self.features.demucs_available,
            'midi_extraction': self.features.music21_available,
            'key_detection': self.features.essentia_available,
            'chord_analysis': self.features.essentia_available,
        }
        
        return feature_requirements.get(feature_name, True)  # Default to available
    
    def print_status(self):
        """Print configuration status for debugging."""
        print("=" * 60)
        print("MusicTruth Configuration Status")
        print("=" * 60)
        print("\nLibrary Availability:")
        print(f"  ✓ librosa: {self.features.librosa_available}")
        print(f"  {'✓' if self.features.audioflux_available else '✗'} audioFlux: {self.features.audioflux_available}")
        print(f"  {'✓' if self.features.essentia_available else '✗'} Essentia: {self.features.essentia_available}")
        print(f"  {'✓' if self.features.torch_available else '✗'} PyTorch: {self.features.torch_available}")
        print(f"  {'✓' if self.features.transformers_available else '✗'} Transformers: {self.features.transformers_available}")
        print(f"  {'✓' if self.features.demucs_available else '✗'} Demucs: {self.features.demucs_available}")
        print(f"  {'✓' if self.features.music21_available else '✗'} music21: {self.features.music21_available}")
        print(f"  {'✓' if self.features.plotly_available else '✗'} Plotly: {self.features.plotly_available}")
        print("\nPaths:")
        print(f"  Input: {self.paths.input_dir}")
        print(f"  Output: {self.paths.output_dir}")
        print(f"  Cache: {self.paths.cache_dir}")
        print("=" * 60)


# Create global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_status()
