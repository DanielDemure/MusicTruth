# 🎵 MusicTruth - AI Music Analysis Tool

**MusicTruth** is a powerful forensic audio analysis tool designed to detect potential AI-generated music. It analyzes audio files for spectral anomalies, robotic timing, and specific artifacts associated with generative AI models like Suno and Udio.

## 🚀 Features

- **Multi-Layered Analysis**:
  - **Spectral Forensics**: Detects hard frequency cutoffs and "grid" artifacts common in AI upsampling.
  - **Tempo Stability**: Analyzes beat consistency to detect robotic quantization.
  - **Stereo Imaging**: Checks for phase coherence and unnatural stereo width.
- **Deep Forensics** (Optional):
  - **Source Separation**: Uses `demucs` to isolate vocals from the mix.
  - **Vocal Analysis**: Checks isolated vocals for "perfect" pitch quantization (Auto-Tune effect) and lack of natural breaths.
- **Album Consistency**: Compares multiple tracks to verify if they share the same sonic fingerprint (singer, production style).
- **Provider Fingerprinting**: Heuristics to identify artifacts specific to **Suno** (high-end sheen) and **Udio** (spectral chirps).
- **Comprehensive Reporting**: Generates detailed Markdown and PDF reports with metrics and flags.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MusicTruth.git
   cd MusicTruth
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For Deep Analysis features, you must ensure `demucs`, `torch`, and `transformers` are installed.*

3. **Install FFmpeg**:
   - **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## 🛠️ Usage

1. **Add Audio Files**:
   - Place your `.mp3`, `.wav`, or `.flac` files in the `input/` directory.
   - Or add URLs to `sources.txt` to download them automatically.

2. **Run the Tool**:
   ```bash
   python src/main.py
   ```

3. **Follow the Prompts**:
   - Select files to analyze.
   - Choose whether to perform **Deep Analysis** (slower, but more accurate) and **Album Consistency Checks**.

4. **View Reports**:
   - Check the `output/` directory for `report.md` and `report.pdf`.

## 🧩 Modules

- `src/main.py`: CLI entry point.
- `src/analyzer.py`: Core analysis logic (Spectral, Tempo, ML).
- `src/separator.py`: Source separation using Demucs.
- `src/comparator.py`: Album consistency and similarity matrix.
- `src/reporter.py`: Report generation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool provides probabilistic scores based on heuristics and known artifacts. It is not a definitive proof of AI generation. False positives are possible, especially with highly processed human music.
