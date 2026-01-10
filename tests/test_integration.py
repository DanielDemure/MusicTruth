"""
Integration test for MusicTruth full pipeline.

Tests: Input -> Process -> Analyze -> Report
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path

# We'll need to generate a test audio file
import numpy as np
import librosa
import soundfile as sf


class TestFullPipeline(unittest.TestCase):
    """Test the complete MusicTruth analysis pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test audio file."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_audio_path = os.path.join(cls.test_dir, "test_audio.wav")
        
        # Generate 5 seconds of test audio (sine wave)
        sr = 22050
        duration = 5
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV
        sf.write(cls.test_audio_path, y, sr)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_analyzer_pipeline(self):
        """Test that Analyzer can process a file end-to-end."""
        from src.layers.analysis.core import Analyzer
        from src.config import AnalysisMode
        
        analyzer = Analyzer()
        
        # Run analysis in QUICK mode
        results = analyzer.analyze_audio(self.test_audio_path, mode=AnalysisMode.QUICK)
        
        # Verify structure
        self.assertIn('filename', results)
        self.assertIn('ai_probability', results)
        self.assertIn('features', results)
        self.assertIsInstance(results['ai_probability'], float)
        self.assertGreaterEqual(results['ai_probability'], 0.0)
        self.assertLessEqual(results['ai_probability'], 1.0)
        
    def test_reporter_generation(self):
        """Test that Reporter can generate HTML."""
        from src.layers.reporting.generator import MultiFormatReporter
        
        # Create output dir
        output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        reporter = MultiFormatReporter(output_dir)
        
        # Mock results
        results = {
            'filename': 'test_audio.wav',
            'ai_probability': 0.35,
            'flags': ['Test flag 1', 'Test flag 2'],
            'features': {}
        }
        
        # Generate report
        reporter.generate(results, output_formats=['html', 'json'])
        
        # Verify files exist
        html_files = list(Path(output_dir).glob("*.html"))
        json_files = list(Path(output_dir).glob("*.json"))
        
        self.assertGreater(len(html_files), 0, "HTML report not generated")
        self.assertGreater(len(json_files), 0, "JSON report not generated")
        
    def test_input_handler(self):
        """Test InputHandler can scan files."""
        from src.layers.input.handler import InputHandler
        
        handler = InputHandler(self.test_dir)
        
        # Add test file
        handler.add_sources_from_paths([self.test_audio_path])
        
        sources = handler.get_ready_sources_list()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_type, 'file')


if __name__ == '__main__':
    unittest.main()
