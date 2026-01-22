import os
import logging
from pathlib import Path
from typing import Dict
from src.utils.logger import logger

def separate_audio(file_path: str, output_dir: str = "temp_separated", model_name: str = "UVR-MDX-Net-Inst_HQ_3.onnx") -> Dict[str, str]:
    """
    Separates audio into stems using audio-separator (supports UVR and Demucs).
    Default model is a high-quality UVR MDX model for instrumental/vocal separation.
    
    Args:
        file_path: Path to the input audio file.
        output_dir: Directory to save separated files.
        model_name: Model to use (default: UVR-MDX-Net-Inst_HQ_3.onnx).
                    Other options: 'htdemucs', 'htdemucs_ft', 'kim_vocal_2'.
    
    Returns:
        Dictionary mapping stem names ('vocals', 'instrumental', etc.) to file paths.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        logger.error("audio-separator not installed. Please install it via pip.")
        return {}

    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Separating stems for {file_path.name} using {model_name}...")
    
    try:
        # Initialize Separator with options
        separator = Separator(
            log_level=logging.WARNING,
            output_dir=output_dir,
            output_format="wav",
            normalization_threshold=0.9
        )
        
        # Load model and separate
        separator.load_model(model_filename=model_name)
        
        # output_files is a list of filenames (not full paths)
        output_files = separator.separate(file_path)
        
        # Map outputs to standard keys
        stems = {}
        
        for f in output_files:
            full_path = output_dir / f
            lower_name = f.lower()
            
            if "vocals" in lower_name:
                stems["vocals"] = str(full_path)
            elif "instrumental" in lower_name or "no_vocals" in lower_name:
                stems["instrumental"] = str(full_path)
            elif "drums" in lower_name:
                stems["drums"] = str(full_path)
            elif "bass" in lower_name:
                stems["bass"] = str(full_path)
            elif "other" in lower_name:
                stems["other"] = str(full_path)
            else:
                # Fallback for unknown stems
                stems[f"stem_{f}"] = str(full_path)
                
        # Basic check for UVR 2-stem models
        if 'vocals' in stems and 'instrumental' not in stems:
             # Sometimes instrumental is named differently or we just need to find it
             pass

        logger.info(f"Separation complete: {list(stems.keys())}")
        return stems

    except Exception as e:
        logger.exception(f"Error during separation: {e}")
        return {}
