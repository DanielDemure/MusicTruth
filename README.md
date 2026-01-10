# 🎵 MusicTruth 2.0 - AI Music Forensics

**MusicTruth** is an advanced open-source tool for detecting AI-generated music. It combines forensic signal processing with modern Large Language Models (LLMs) to provide definitive, human-readable authenticity reports.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-green)

---

## 🚀 Key Features

### 🧠 **AI Intelligence**

* **Universal LLM Support**: Connect your preferred AI brain:
  * **Commercial**: OpenAI (GPT-4), Anthropic (Claude 3.5), Google (Gemini 1.5), DeepSeek.
  * **Local / Private**: Ollama, LM Studio (OpenAI-compatible).
  * **Aggregators**: OpenRouter.
* **Three-Agent System**:
    1. **Researcher**: Finds artist context and discography.
    2. **Critic**: Reviews technical metrics against the artist's known style.
    3. **Reporter**: Writes the final public-facing verdict.

### 🕵️ **Forensic Analysis**

* **19+ Feature Extractors**:
  * **Spectral**: hard cutoffs (16kHz/20kHz), grid anomalies, unnatural silence.
  * **Temporal**: Robotic quantization, perfect beat stability.
  * **Vocal**: "Perfect pitch" (Auto-tune) artifacts, lack of breath.
  * **Provider Fingerprints**: Detects specific artifacts from **Suno** and **Udio**.
* **Source Separation**: Isolates vocals/drums using `demucs` for cleaner analysis.

### 📡 **Universal Input**

* **Interactive Wizard**: User-friendly TUI to configure your run.
* **Remote Downloads**:
  * **Spotify**: Automatically fetches tracks with metadata via `spotdl`.
  * **YouTube**: High-quality rips via `yt-dlp`.
* **Multi-Source Comparison**: Add file + URL versions of the same song to separate encoding artifacts from AI generation errors.

---

## 📦 Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/DanielDemure/MusicTruth.git
    cd MusicTruth
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Includes `spotdl` and `yt-dlp` for functionality.*

3. **Install FFmpeg**:
    * **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add to PATH.
    * **Mac**: `brew install ffmpeg`

3.  **Install FFmpeg**:
    *   **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add to PATH.
    *   **Mac**: `brew install ffmpeg`

4.  **(Optional) Install Deepfake/Transcription/PDF Support**:

    ```bash
    # For Deepfake Detection
    pip install torch
    
    # For MIDI Transcription (Optional, Blocked on Python 3.12 due to TensorFlow conflict)
    # pip install basic-pitch
    
    # For PDF Reporting (Optional, Blocked on Python 3.12)
    # pip install weasyprint
    ```

---

## 🛠️ Usage

### 🧙 **Interactive Mode (Recommended)**

Simply run the tool without arguments to launch the Wizard:

```bash
python -m src.main
```

The wizard will guide you through:

1.  **Project Name**: Organize your results.
2.  **Inputs**: Add Folders, Files, or **Spotify/YouTube URLs**.
3.  **Mode**: Choose from Quick (30s) to Forensic (30m+).
4.  **LLM Config**: Select your provider (e.g. Gemini, OpenAI) and model.

### 💻 **CLI Mode (Advanced)**

For automation or power users:

```bash
python -m src.main \
  --input "https://open.spotify.com/track/..." \
  --project "Analysis_V1" \
  --mode forensic \
  --llm-provider openai \
  --llm-key "sk-..." 
```

---

## 🧩 Architecture (v2 Upgrade)
The codebase follows a reliable 5-Layer Agentic Architecture:

1.  **Input Layer** (`src/layers/input`): Handles file scanning, URL downloading (`spotdl`, `yt-dlp`), and source grouping.
2.  **Processing Layer** (`src/layers/processing`): Heavy lifting like Source Separation (`audio-separator`/Demucs/UVR).
3.  **Analysis Layer** (`src/layers/analysis`):
    *   **Core**: Orchestrates extractors.
    *   **Features**: Modular extractors (Bio-inspired, Spectral, Temporal, Essentia, Transcription).
    *   **Detectors**: AI/ML models (Deepfake, Fingerprinting).
4.  **Reporting Layer** (`src/layers/reporting`): Generates HTML/PDF reports using Jinja2 templates.
5.  **Orchestration Layer** (`src/layers/orchestration`): Manages the LLM Agents (Critic, Researcher, Reporter) and session history.

## ⚠️ Disclaimer

MusicTruth provides probabilistic scores based on technical heuristics. False positives are possible with highly processed human electronic music. Always verify with human judgment.
