"""
Visualization generators for MusicTruth reports.

Creates interactive Plotly visualizations for audio analysis.
"""

import numpy as np
import librosa
import librosa.display
from typing import Optional, Dict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def generate_spectrogram_plot(audio_path: str, y: Optional[np.ndarray] = None, 
                              sr: Optional[int] = None) -> Optional[str]:
    """
    Generate an interactive Plotly spectrogram.
    
    Returns:
        HTML div string for embedding, or None if failed
    """
    if not PLOTLY_AVAILABLE:
        return None
        
    try:
        # Load audio if not provided
        if y is None or sr is None:
            y, sr = librosa.load(audio_path, sr=22050)
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Create Plotly heatmap
        times = librosa.times_like(D, sr=sr)
        freqs = librosa.fft_frequencies(sr=sr)
        
        fig = go.Figure(data=go.Heatmap(
            z=D,
            x=times,
            y=freqs,
            colorscale='Viridis',
            colorbar=dict(title='dB')
        ))
        
        fig.update_layout(
            title='Spectrogram',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Return as div (no full HTML wrapper)
        return fig.to_html(include_plotlyjs='cdn', div_id='spectrogram', full_html=False)
        
    except Exception as e:
        print(f"⚠️ Failed to generate spectrogram: {e}")
        return None


def generate_feature_radar_chart(features: Dict) -> Optional[str]:
    """
    Generate a radar chart of feature scores.
    
    Args:
        features: Dict of feature_name -> {score, confidence}
    
    Returns:
        HTML div string or None
    """
    if not PLOTLY_AVAILABLE:
        return None
        
    try:
        # Extract scores
        categories = []
        scores = []
        
        for name, data in features.items():
            if isinstance(data, dict) and 'score' in data:
                categories.append(name.replace('_', ' ').title())
                scores.append(data['score'] * 100)  # Convert to percentage
        
        if not categories:
            return None
        
        # Create radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='AI Suspicion Score'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title='Feature Analysis',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='radar-chart', full_html=False)
        
    except Exception as e:
        print(f"⚠️ Failed to generate radar chart: {e}")
        return None
