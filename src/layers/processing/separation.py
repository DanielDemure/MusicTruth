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
    if "demucs" in model_name:
        return separate_audio_demucs(file_path, output_dir, model_name)

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


def separate_audio_demucs(file_path: str, output_dir: str = "temp_separated", model_name: str = "htdemucs_ft") -> Dict[str, str]:
    """
    Separates audio using Demucs explicitly (subprocess or library).
    Preferred for forensic analysis due to better quality on 'other' (piano) stem.
    
    Args:
        file_path: Path to input audio
        output_dir: Base output directory
        model_name: Demucs model code (htdemucs_ft, htdemucs_6s, etc.)
        
    Returns:
        Dict mapping stem names to absolute file paths.
    """
    import subprocess
    import shutil
    
    file_path = Path(file_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    logger.info(f"Starting Demucs separation for {file_path.name} (model: {model_name})...")
    
    # Construct command: demucs -n <model> -o <outdir> <file>
    # We use subprocess to isolate it and capture output
    cmd = [
        "demucs",
        "-n", model_name,
        "-o", str(output_dir),
        str(file_path)
    ]
    
    try:
        # Check if installed
        subprocess.run(["demucs", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Demucs CLI not found. Please install with 'pip install demucs'.")
        return {}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # logger.debug(f"Demucs output: {result.stdout}")
        
        # Demucs output structure: <outdir>/<model>/<track_name>/<stem>.wav
        # We need to find this folder. 'track_name' is usually filename without ext.
        track_name = file_path.stem
        
        # Demucs might sanitize the filename, so we look for the directory
        expected_dir = output_dir / model_name / track_name
        
        if not expected_dir.exists():
            # Try to find it broadly if name sanitization happened
            start_dir = output_dir / model_name
            candidates = list(start_dir.glob("*"))
            # Heuristic: find most recent directory or matching name
            if candidates:
                expected_dir = candidates[0] # Assumption if running singly
                logger.warning(f"Could not find exact match for {track_name}, using {expected_dir.name}")
            else:
                logger.error(f"Demucs finished but output directory empty: {start_dir}")
                return {}
                
        # Map stems
        stems = {}
        for stem_file in expected_dir.glob("*.wav"):
            key = stem_file.stem # vocals, drums, bass, other
            stems[key] = str(stem_file)
            
        logger.info(f"Demucs separation successful. Stems: {list(stems.keys())}")
        return stems
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs separation failed: {e.stderr}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error in Demucs separation: {e}")
        return {}
