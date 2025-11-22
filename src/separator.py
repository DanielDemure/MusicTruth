import os
import subprocess
import shutil
from pathlib import Path
import logging

def separate_audio(file_path: str, output_dir: str = "temp_separated") -> dict:
    """
    Separates audio into stems (vocals, drums, bass, other) using Demucs.
    Returns a dictionary of paths to the separated files.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Separating stems for {file_path.name} (this may take a while)...")
    
    # Construct command
    # Try to find demucs executable
    demucs_cmd = "demucs"
    
    # Check if demucs is in PATH
    if shutil.which("demucs") is None:
        # Check common Windows user script path
        import sys
        scripts_dir = Path(sys.executable).parent / "Scripts"
        # Or user roaming path
        user_scripts = Path(os.path.expanduser("~")) / "AppData/Roaming/Python/Python312/Scripts"
        
        if (scripts_dir / "demucs.exe").exists():
            demucs_cmd = str(scripts_dir / "demucs.exe")
        elif (user_scripts / "demucs.exe").exists():
            demucs_cmd = str(user_scripts / "demucs.exe")
        else:
            # Fallback: try running as module (if supported) or just hope
            pass

    cmd = [
        demucs_cmd,
        "-n", "htdemucs", 
        "--out", str(output_dir),
        str(file_path)
    ]
    
    try:
        # Run demucs via subprocess to avoid complex dependency issues in-process
        # and to capture output easily.
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Demucs failed: {result.stderr}")
            return {}
            
        # Demucs output structure: output_dir / model_name / track_name / stem.wav
        # e.g. temp_separated/htdemucs/song_name/vocals.wav
        
        # Need to find the actual output folder. Demucs cleans the filename.
        # We can look for the folder inside output_dir/htdemucs
        model_dir = output_dir / "htdemucs"
        
        # Find the folder that matches the file stem (ignoring extension)
        # Demucs might replace spaces with underscores etc.
        # Simplest way: list dirs in model_dir and find the newest one or match name
        
        expected_name = file_path.stem
        track_dir = model_dir / expected_name
        
        if not track_dir.exists():
            # Try to find it loosely
            for d in model_dir.iterdir():
                if d.is_dir():
                    # If we just processed one file, it's likely this one.
                    # But for safety, let's assume it matches somewhat.
                    pass
            
            # If still not found, return empty
            if not track_dir.exists():
                 print(f"Could not locate output directory for {expected_name}")
                 return {}

        stems = {
            'vocals': str(track_dir / "vocals.wav"),
            'drums': str(track_dir / "drums.wav"),
            'bass': str(track_dir / "bass.wav"),
            'other': str(track_dir / "other.wav")
        }
        
        # Verify they exist
        if not os.path.exists(stems['vocals']):
            print("Vocals stem not found.")
            return {}
            
        return stems

    except FileNotFoundError:
        print("Demucs not installed or not in PATH. Please run 'pip install demucs' and ensure it's accessible.")
        return {}
    except Exception as e:
        print(f"Error during separation: {e}")
        return {}
