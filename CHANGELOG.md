# Changelog

All notable changes to MusicTruth will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-10

### 🎉 Major Release: Complete Rewrite & Enhancement

This release represents a complete architectural overhaul of MusicTruth with extensive new features for AI music detection and forensic analysis.

### Added

#### 🧙 Interactive Wizard System
- **10-Step Interactive Workflow**: Comprehensive guided setup for analysis projects
- **Metadata Auto-Fetching**: Automatic enrichment from MusicBrainz and Spotify APIs
- **Interactive Metadata Review**: Edit and correct artist/album/track information before analysis
- **Pre-flight System Validation**: Comprehensive checks for API keys, disk space, and file accessibility
- **Back Navigation**: Full support for undoing and re-navigating wizard steps
- **`.env` Auto-Detection**: Automatically loads API credentials from environment files

#### 📊 Advanced Reporting System
- **5 Report Styles**: 
  - Technical (raw metrics and spectral data)
  - Human-Readable (plain language findings)
  - Executive Summary (1-page overview)
  - Combined (technical + summary)
  - Forensic Expert (everything + raw JSON export)
- **Multi-Format Output**: HTML, JSON, PDF (via WeasyPrint), CSV
- **Interactive Visualizations**: Plotly-powered spectrograms and radar charts
- **Jinja2 Templating**: Customizable report templates

#### 🤖 AI Music Detection
- **Hugging Face Transformers Integration**: Using `AI-Music-Detection/ai_music_detection_large_60s` model
- **GPU Acceleration**: Automatic detection and usage of available CUDA devices
- **Reliable Model Lifecycle**: Auto-download and caching from Hugging Face Hub
- **High Accuracy**: Optimized for detecting Suno, Udio, and other modern AI music generators

#### 🏗️ New Architecture
- **Layered Design**: Separation of concerns across Input, Processing, Analysis, Orchestration, and Reporting layers
- **LLM Integration**: Multi-provider support (OpenAI, Anthropic, Google Gemini, DeepSeek, Ollama, Custom)
- **AI Agents**: Specialized agents for research, critique, and public reporting
- **History Management**: Project-based session tracking with artist/album folder organization

#### 🎵 Enhanced Analysis Features
- **Multi-Source Verification**: Compare different versions of the same track
- **Group ID System**: Organize and cross-check related audio sources
- **Advanced Spectral Analysis**: Comprehensive frequency domain analysis
- **Tempo & Beat Detection**: Rhythm pattern analysis
- **Source Separation**: Optional stem extraction for deeper analysis (DeepSeek/Spleeter)
- **MIDI Transcription**: Audio-to-MIDI conversion (optional, requires Python <3.12)

#### 📁 Advanced Output Organization
- **Artist/Album Folder Structure**: Automatic organization as `output/Artist/Album/Timestamp/`
- **Sanitized Naming**: Cross-platform compatible folder names
- **Session Versioning**: Timestamped session directories for history tracking

### Changed

- **Complete Restructure**: Migrated from flat `src/` to layered `src/layers/` architecture
- **Dependency Management**: Updated to support Python 3.12 with selective optional dependencies
- **Configuration System**: Centralized config with `.env` support via `python-dotenv`
- **Input Handling**: Enhanced with multi-source support and remote URL downloading

### Deprecated

- **Old Flat Architecture**: Legacy files moved/refactored into new layer structure
- **Deezer Detector**: Replaced with Hugging Face AI Music Detector

### Removed

- **Python 3.12 Incompatible Strict Dependencies**: 
  - `basic-pitch` (MIDI transcription - now optional)
  - `weasyprint` (PDF export - now optional)

### Fixed

- **Package Naming**: Corrected `python-audio-separator` to `audio-separator`
- **NumPy 2.0 Compatibility**: Pinned to `<2.0.0` for stability
- **Duplicate Imports**: Cleaned up `src/main.py`
- **Virtual Environment Conflicts**: Documented resolution for `spotdl`/`mcp` conflicts

### Documentation

- **Comprehensive README**: Updated with new architecture, features, and installation guides
- **Setup Guides**: 
  - `.env` configuration guide (`directives/env_setup_guide.md`)
  - AI Music Detection setup (`directives/ai_detection_setup.md`)
- **Walkthrough Document**: Complete feature overview and verification report
- **API Documentation**: Inline docstrings for all major modules

### Dependencies

#### New Required Dependencies
- `mutagen>=1.47.0` - Metadata extraction
- `musicbrainzngs>=0.7.1` - MusicBrainz API integration
- `spotipy>=2.23.0` - Spotify API integration
- `python-dotenv>=1.0.0` - Environment variable management
- `transformers>=4.35.0` - Hugging Face model support
- `rich>=13.7.0` - Terminal UI
- `questionary>=2.0.1` - Interactive prompts

#### Optional Dependencies
- `basic-pitch` - MIDI transcription (Python <3.12 only)
- `weasyprint` - PDF export (Python <3.12 only)

### Security

- **`.env` Protection**: Automatic `.gitignore` rules for environment files
- **API Key Masking**: Credentials never printed in logs
- **Secure Credential Storage**: Documentation on best practices for API key management

---

## [1.0.0] - 2024-12-XX (Historical)

### Initial Release
- Basic audio analysis functionality
- Simple CLI interface
- Limited reporting options

