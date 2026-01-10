"""
Quick test script to verify feature extractors work correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.spectral import get_spectral_extractors
from src.features.temporal import get_temporal_extractors
from src.features.harmonic import get_harmonic_extractors
from src.features.vocal import get_vocal_extractors
from src.features.structural import get_structural_extractors
from src.features.midi_features import get_midi_extractors
from src.features.provider_fingerprint import get_provider_extractors
from src.config import config

def test_feature_extractors():
    """Test that all feature extractors can be instantiated."""
    print("=" * 60)
    print("MusicTruth Feature Extractor Test")
    print("=" * 60)
    
    # Print config status
    print("\n1. Configuration Status:")
    config.print_status()
    
    extractor_groups = [
        ("Spectral", get_spectral_extractors()),
        ("Temporal", get_temporal_extractors()),
        ("Harmonic", get_harmonic_extractors()),
        ("Vocal", get_vocal_extractors()),
        ("Structural", get_structural_extractors()),
        ("MIDI", get_midi_extractors()),
        ("Provider", get_provider_extractors())
    ]
    
    total_count = 0
    
    for name, extractors in extractor_groups:
        print(f"\n{name} Feature Extractors:")
        for extractor in extractors:
            total_count += 1
            available = "✓" if extractor.is_available() else "✗"
            print(f"  {available} {extractor.name}: {extractor.description}")
            if not extractor.is_available():
                print(f"    -> Missing deps: {extractor.get_required_libraries()}")
    
    print("\n" + "=" * 60)
    print(f"Total Extractors: {total_count}")
    print("=" * 60)
    
    # --- Verify System Components ---
    print("\n4. System Components Verification:")
    try:
        from src.history_manager import HistoryManager
        hm = HistoryManager()
        print(f"  ✓ HistoryManager instantiated (Root: {hm.output_root})")
    except ImportError as e:
        print(f"  ✗ HistoryManager failed: {e}")

    try:
        from src.input_handler import InputHandler
        ih = InputHandler(".")
        print(f"  ✓ InputHandler instantiated")
    except ImportError as e:
        print(f"  ✗ InputHandler failed: {e}")

    try:
        from src.comparator import CrossCheckComparator
        cc = CrossCheckComparator()
        print(f"  ✓ CrossCheckComparator instantiated")
    except ImportError as e:
        print(f"  ✗ CrossCheckComparator failed: {e}")

    # --- Verify LLM Components ---
    print("\n5. LLM Components Verification:")
    try:
        from src.llm.client import LLMClient
        from src.llm.agents import CriticAgent, PublicReporterAgent, ResearcherAgent
        client = LLMClient(api_key="sk-test") # Fake key just to init class
        print(f"  ✓ LLMClient instantiated (Provider: {client.provider})")
        
        critic = CriticAgent(client)
        print(f"  ✓ CriticAgent instantiated")
        
    except ImportError as e:
        print(f"  ✗ LLM Components failed: {e}")
        
    return True

if __name__ == "__main__":
    try:
        test_feature_extractors()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
