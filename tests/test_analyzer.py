import unittest
import numpy as np
import librosa
from src.analyzer import check_frequency_cutoff, check_tempo_stability

class TestAnalyzer(unittest.TestCase):
    def test_frequency_cutoff_synthetic(self):
        # Generate a signal with a hard cutoff at 16kHz
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # White noise
        y = np.random.normal(0, 1, len(t))
        
        # Low-pass filter at 16kHz
        # We use a simple FFT manipulation for testing
        D = librosa.stft(y)
        freqs = librosa.fft_frequencies(sr=sr)
        D[freqs > 16000] = 0
        y_filtered = librosa.istft(D)
        
        score, cutoff = check_frequency_cutoff(y_filtered, sr)
        
        print(f"Detected cutoff: {cutoff} Hz, Score: {score}")
        # Spectral leakage might push it slightly above 16000
        self.assertTrue(15000 < cutoff < 18000)
        self.assertTrue(score > 0.3)

    def test_tempo_stability_synthetic(self):
        # Generate a perfectly quantized beat
        sr = 22050
        duration = 5.0
        bpm = 120
        # Generate click times
        times = np.arange(0, duration, 60.0/bpm)
        y = librosa.clicks(times=times, sr=sr, length=int(duration*sr))
        
        score, bpm, cv = check_tempo_stability(y, sr)
        print(f"Detected BPM: {bpm}, Stability Score: {score}, CV: {cv}")
        
        # Should be very stable -> high score
        self.assertTrue(score > 0.5)

    def test_spectral_peaks_synthetic(self):
        # Generate a signal with a "comb" spectrum (simulating deconvolution artifacts)
        sr = 44100
        duration = 1.0
        y = np.zeros(int(sr * duration))
        
        # Add sine waves at regular intervals
        t = np.linspace(0, duration, len(y))
        for f in range(10000, 20000, 500):
            y += 0.1 * np.sin(2 * np.pi * f * t)
            
        from src.analyzer import check_spectral_peaks
        score, var = check_spectral_peaks(y, sr)
        print(f"Spectral Peak Score: {score}, Variance: {var}")
        self.assertTrue(score > 0.0)

    def test_stereo_analysis_synthetic(self):
        # Generate a mono signal in stereo container
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y_mono = np.sin(2 * np.pi * 440 * t)
        y_stereo = np.vstack([y_mono, y_mono]) # Identical channels
        
        from src.analyzer import check_stereo_analysis
        score, ratio = check_stereo_analysis(y_stereo)
        print(f"Stereo Score: {score}, Ratio: {ratio}")
        
        # Ratio should be 0 (perfect mono)
        # Score should be 0.2 (weak indicator)
        self.assertAlmostEqual(ratio, 0.0)
        self.assertEqual(score, 0.2)

if __name__ == '__main__':
    unittest.main()
