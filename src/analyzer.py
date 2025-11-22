import librosa
import numpy as np
import scipy.stats
from typing import Dict, Any, List
import warnings

# Try importing torch/transformers, but don't fail if missing (allow running without ML)
try:
    import torch
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: torch or transformers not found. ML analysis will be skipped.")

def analyze_audio(file_path: str) -> Dict[str, Any]:
    """
    Performs a comprehensive analysis of the audio file.
    Returns a dictionary containing metrics and an AI probability score.
    """
    print(f"Analyzing {file_path}...")
    results = {
        'filename': file_path,
        'duration': 0,
        'sample_rate': 0,
        'ai_probability': 0.0,
        'flags': [],
        'metrics': {}
    }

    try:
        # Load audio
        # We load with native sampling rate to detect cutoffs correctly
        y, sr = librosa.load(file_path, sr=None)
        results['duration'] = librosa.get_duration(y=y, sr=sr)
        results['sample_rate'] = sr

        # 1. Spectral Analysis (Frequency Cutoff)
        cutoff_score, cutoff_freq = check_frequency_cutoff(y, sr)
        results['metrics']['spectral_cutoff'] = cutoff_freq
        if cutoff_score > 0.5:
            results['flags'].append(f"Hard frequency cutoff detected at {cutoff_freq/1000:.1f}kHz")
            results['ai_probability'] += 0.3  # Contribution to probability

        # 2. Tempo Stability / Quantization
        tempo_score, bpm, cv = check_tempo_stability(y, sr)
        results['metrics']['bpm'] = bpm
        results['metrics']['tempo_stability_score'] = tempo_score
        results['metrics']['tempo_cv'] = cv
        
        if tempo_score > 0.5:
            results['flags'].append(f"High tempo stability (CV: {cv:.4f})")
            results['ai_probability'] += 0.2 * tempo_score

        # 3. Silence / Noise Floor Analysis
        silence_score = check_silence_anomalies(y)
        if silence_score > 0.5:
            results['flags'].append("Anomalous digital silence detected")
            results['ai_probability'] += 0.1

        # 4. Advanced Spectral Analysis (Grid/Peaks)
        peak_score, peak_var = check_spectral_peaks(y, sr)
        results['metrics']['spectral_peak_var'] = peak_var
        if peak_score > 0.5:
            results['flags'].append(f"Systematic spectral peaks detected (Var: {peak_var:.2e})")
            results['ai_probability'] += 0.3 * peak_score

        # 5. Stereo Analysis
        stereo_score, stereo_ratio = check_stereo_analysis(y)
        results['metrics']['stereo_ratio'] = stereo_ratio
        if stereo_score > 0.5:
            results['flags'].append(f"Suspicious stereo imaging (Ratio: {stereo_ratio:.2f})")
            results['ai_probability'] += 0.1 * stereo_score

        # 6. Machine Learning Check (if available)
        if ML_AVAILABLE:
            ml_score, ml_label = check_ai_model(file_path)
            results['metrics']['ml_score'] = ml_score
            results['metrics']['ml_label'] = ml_label
            if ml_score > 0.5:
                results['flags'].append(f"ML Model detected {ml_label} ({ml_score:.2f})")
                # ML is usually high confidence, so we weight it heavily
                results['ai_probability'] += 0.4 * ml_score

        # Normalize probability
        results['ai_probability'] = min(1.0, results['ai_probability'])

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        results['error'] = str(e)

    return results

def check_frequency_cutoff(y: np.ndarray, sr: int) -> (float, float):
    """
    Checks for hard frequency cutoffs in the spectrogram.
    Returns a score (0-1) and the detected cutoff frequency.
    """
    # Use spectral rolloff to find the effective cutoff frequency
    # We use a high percentile (e.g. 99%) to find where the energy really drops off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    cutoff_freq = np.mean(rolloff)
    
    score = 0.0
    
    # Continuous scoring logic
    # 16kHz is a common cutoff for older AI (and 128kbps MP3)
    # 22.05kHz is Nyquist for 44.1kHz
    # 24kHz is Nyquist for 48kHz
    
    # If cutoff is very low (< 10kHz), it's very poor quality (score 1.0)
    # If cutoff is around 16kHz (15-17k), score high (0.8)
    # If cutoff is around 20kHz (19-21k), score medium (0.4)
    # If cutoff is > 21k, score low (0.0)
    
    if cutoff_freq < 10000:
        score = 0.9
    elif cutoff_freq > 21000:
        score = 0.0
    else:
        # Gaussian-like bell curves around suspicious frequencies
        # Peak at 16000
        dist_16k = abs(cutoff_freq - 16000)
        score_16k = 0.8 * np.exp(-(dist_16k**2) / (2 * 1000**2)) # Width of 1000Hz
        
        # Peak at 20000
        dist_20k = abs(cutoff_freq - 20000)
        score_20k = 0.4 * np.exp(-(dist_20k**2) / (2 * 1000**2))
        
        score = max(score_16k, score_20k)
        
    return score, cutoff_freq

def check_tempo_stability(y: np.ndarray, sr: int) -> (float, float, float):
    """
    Analyzes beat consistency. 
    Returns a stability score (0-1), BPM, and CV.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # librosa.beat.beat_track returns a scalar tempo in newer versions, but sometimes an array?
    # Ensure tempo is a float
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]

    if len(beats) < 2:
        return 0.0, tempo, 0.0
        
    beat_times = librosa.frames_to_time(beats, sr=sr)
    intervals = np.diff(beat_times)
    
    if len(intervals) == 0:
        return 0.0, tempo, 0.0

    std_dev = np.std(intervals)
    avg_interval = np.mean(intervals)
    
    if avg_interval == 0:
        return 0.0, tempo, 0.0

    cv = std_dev / avg_interval 
    
    # Continuous scoring for stability
    # CV < 0.01 is very robotic (score 1.0)
    # CV > 0.05 is very human (score 0.0)
    
    # Linear interpolation between 0.01 and 0.05
    # score = 1.0 - (cv - 0.01) / (0.04)
    
    if cv < 0.01:
        score = 0.9
    elif cv > 0.05:
        score = 0.0
    else:
        score = 0.9 * (1.0 - (cv - 0.01) / 0.04)
        
    return score, tempo, cv

def check_silence_anomalies(y: np.ndarray) -> float:
    """
    Checks for absolute digital zeros which are rare in analog/human recordings.
    """
    # Check for exact zeros
    zero_count = np.sum(y == 0)
    total_samples = len(y)
    
    # If we have significant chunks of exact zeros, it might be digital splicing or generation
    # But silence at start/end is normal.
    
    # This is a weak heuristic, just a placeholder for now.
    return 0.0

def check_spectral_peaks(y: np.ndarray, sr: int) -> (float, float):
    """
    Checks for systematic spectral peaks (grid artifacts) often caused by 
    deconvolution layers in GANs/Diffusion models.
    Returns a score (0-1) and the peak variance.
    """
    # Compute STFT
    D = np.abs(librosa.stft(y))
    mean_spectrum = np.mean(D, axis=1)
    
    # Look at high frequencies where artifacts are most visible (> 10kHz)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = freqs > 10000
    if not np.any(mask):
        return 0.0, 0.0
        
    high_freq_spectrum = mean_spectrum[mask]
    
    # Calculate the "spikiness" or variance of the derivative
    # Natural noise floor is usually smooth. 
    # Deconvolution artifacts look like a comb filter or regular grid.
    
    # Normalize
    if np.max(high_freq_spectrum) == 0:
         return 0.0, 0.0
         
    norm_spec = high_freq_spectrum / np.max(high_freq_spectrum)
    
    # Take second derivative to find sharp peaks
    d2 = np.diff(norm_spec, 2)
    peak_variance = np.var(d2)
    
    # Heuristic: High variance in the 2nd derivative of the high-freq spectrum
    # implies unnatural "comb" or "grid" patterns.
    
    # Thresholds need tuning, but let's assume:
    # > 1e-5 is suspicious
    
    score = 0.0
    if peak_variance > 1e-4:
        score = 0.9
    elif peak_variance > 1e-5:
        score = 0.5
    else:
        score = 0.0
        
    return score, peak_variance

def check_stereo_analysis(y: np.ndarray) -> (float, float):
    """
    Analyzes stereo imaging.
    Returns a score (0-1) and the Side/Mid energy ratio.
    """
    if y.ndim < 2:
        # Mono file
        return 0.0, 0.0
        
    # y shape is (2, N) for stereo
    left = y[0]
    right = y[1]
    
    mid = (left + right) / 2
    side = (left - right) / 2
    
    energy_mid = np.sum(mid**2)
    energy_side = np.sum(side**2)
    
    if energy_mid == 0:
        return 0.0, 0.0
        
    ratio = energy_side / energy_mid
    
    # AI often produces "muddy" stereo or effectively mono with phase noise.
    # Very low side energy (near mono) in a "produced" track is suspicious if it's not an old recording.
    # Extremely high side energy (phase issues) is also suspicious.
    
    score = 0.0
    if ratio < 0.01:
        # Almost mono
        score = 0.2 # Weak indicator, could just be mono
    elif ratio > 1.0:
        # More side than mid? Phase cancellation issues.
        score = 0.6
        
    return score, ratio

def check_ai_model(file_path: str) -> (float, str):
    """
    Uses a pre-trained Hugging Face model to detect AI generation.
    Returns a probability score (0-1) and a label.
    """
    if not ML_AVAILABLE:
        return 0.0, "ML_UNAVAILABLE"
        
    try:
        # Use a lightweight model or specific deepfake detector
        # For this MVP, we'll try to load a known model. 
        # Note: This requires internet access to download the model the first time.
        model_id = "MelodyMachine/Deepfake-audio-detection-V2" 
        
        # We use a singleton-like pattern or cache to avoid reloading model every time?
        # For simplicity, we load here (inefficient for batch, but safe).
        # In prod, we'd load this once at module level.
        
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        
        # Load audio and resample to 16kHz (standard for these models)
        y, sr = librosa.load(file_path, sr=16000)
        
        # Preprocess
        # Define max_length (e.g., 30 seconds at 16kHz)
        max_length = 16000 * 30
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=max_length)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probs, dim=-1).item()
        predicted_label = model.config.id2label[predicted_class_id]
        score = probs[0][predicted_class_id].item()
        
        # Map label to AI probability
        # Labels depend on the model. Usually 'fake' or 'real'.
        if predicted_label.lower() in ['fake', 'spoof', 'ai']:
            return score, predicted_label
        else:
            return 0.0, predicted_label
            
    except Exception as e:
        print(f"ML Check failed: {e}")
        return 0.0, "ERROR"

def check_provider_fingerprints(y: np.ndarray, sr: int) -> List[str]:
    """
    Checks for specific artifacts associated with known AI providers.
    """
    flags = []
    
    # Suno: "Hissy" high end / Sheen
    # Check energy > 16kHz relative to total
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    high_freq_mask = freqs > 16000
    
    if np.any(high_freq_mask):
        high_energy = np.sum(S[high_freq_mask, :])
        total_energy = np.sum(S)
        
        if total_energy > 0:
            ratio = high_energy / total_energy
            # Heuristic: suspicious if high frequency energy is oddly high/uniform
            # This is hard to tune generally, but let's look for the "sheen"
            pass 

    # Udio: "Bird chirping" / high freq anomalies
    # Often manifests as transient noise in high bands
    
    return flags

def check_vocal_forensics(vocal_path: str) -> Dict[str, Any]:
    """
    Analyzes isolated vocals for AI artifacts.
    """
    results = {'score': 0.0, 'flags': []}
    
    try:
        y, sr = librosa.load(vocal_path, sr=None)
        
        # 1. Pitch Quantization (Auto-Tune effect)
        # Estimate pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Filter only voiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            # Calculate deviation from nearest semitone
            midi_pitch = librosa.hz_to_midi(f0_voiced)
            nearest_note = np.round(midi_pitch)
            deviation = np.abs(midi_pitch - nearest_note)
            
            avg_deviation = np.mean(deviation)
            
            # AI (and Auto-Tune) tends to have very low deviation (perfect pitch)
            # Natural singing has vibrato and drift (avg dev ~0.1-0.2 semitones?)
            # "Perfect" quantization < 0.05
            
            results['pitch_deviation'] = avg_deviation
            if avg_deviation < 0.05:
                results['score'] += 0.4
                results['flags'].append(f"Unnaturally perfect pitch (Dev: {avg_deviation:.3f})")
        
        # 2. Breath Detection
        # Hard to do without a specific model, but we can look for silence gaps
        # AI often generates continuous singing without breath pauses
        
        return results
        
    except Exception as e:
        print(f"Vocal analysis failed: {e}")
        return results
