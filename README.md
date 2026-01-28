# ⚽ NLP Football Match Analysis Pipeline

An NLP (Natural Language Processing) pipeline that automatically analyzes football match videos. It extracts audio from video files, converts speech to text, and applies various NLP techniques for analysis.

## 🎯 Features

- **Audio Extraction**: Extract audio from video files (ffmpeg)
- **Transcription**: Speech-to-text conversion using OpenAI Whisper
- **Text Preprocessing**: Tokenization, lemmatization, noise removal (spaCy)
- **Information Extraction**: Player, team, stadium, referee detection (NER)
- **Event Detection**: Goal, foul, card, offside, injury detection
- **Sentiment Analysis**: Excitement level and atmosphere analysis
- **Audio Analysis**: Volume level correlation with excitement
- **Summarization**: Abstractive summarization using BART model
- **Visualization**: Excitement graphs, event timelines

## 📁 Project Structure

```
nlpfootbal/
├── main.py                    # Main orchestration
├── extract_audio.py           # Video → WAV
├── transcribe_audio.py        # Audio → Text (Whisper)
├── pre_process.py             # Text preprocessing (spaCy)
├── information_extraction.py  # NER + Event detection
├── sentiment_analyzer.py      # Sentiment/excitement analysis
├── audio_volume.py            # Audio volume analysis
├── summarization.py           # Match summary (BART)
├── insights.py                # Visualization and reporting
├── outputs/                   # Analysis outputs
└── requirements.txt           # Python dependencies
```

## 🚀 Installation

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. System Dependencies

```bash
# ffmpeg installation (Ubuntu/Debian)
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 🎮 Usage

### Single Match Analysis

```python
from main import analyze_match

video_path = "match_video.mp4"
output_dir = analyze_match(video_path)
print(f"Results: {output_dir}")
```

### Command Line

```bash
python main.py
```

> Update the `video` variable in `main.py` with the path to the video you want to analyze.

## 📊 Outputs

A separate folder is created for each match:

| File | Description |
|------|-------------|
| `*_transcript.txt` | Raw transcript |
| `*_preprocessed.txt` | Cleaned text |
| `*_events.json` | Detected events |
| `*_summary.txt` | Structured match summary |
| `*_excitement_graph.png` | Excitement graph |
| `*_event_timeline.png` | Event timeline |
| `*_match_insights.json` | Detailed statistics |
| `*_match_events.csv` | Events in CSV format |

## 🏈 Detected Events

| Event | Pattern Examples |
|-------|------------------|
| ⚽ Goal | "scores!", "it's a goal", "1-0" |
| 🟨 Yellow Card | "yellow card", "booked" |
| 🟥 Red Card | "red card", "sent off" |
| 📐 Offside | "offside", "flag is up" |
| 🔄 Substitution | "substitution", "brings on" |
| 🩹 Injury | "injury", "stretcher" |
| 🦵 Foul | "foul", "tackled" |

## 🔧 Pipeline Flow

```
Video (.mp4)
    ↓
[extract_audio] → Audio (.wav)
    ↓
[transcribe_audio] → Transcript + Timestamps
    ↓
[pre_process] → Cleaned Text
    ↓
[information_extraction] → Entities + Events
    ↓
[sentiment_analyzer] + [audio_volume] → Excitement Analysis
    ↓
[summarization] → Match Summary
    ↓
[insights] → Graphs & Reports
```

## 📦 Technologies

- **spaCy** - NER and linguistic analysis
- **OpenAI Whisper** - Speech recognition
- **Transformers (BART)** - Summarization
- **SciPy/NumPy** - Audio analysis
- **Matplotlib** - Visualization

## 📝 License

MIT License
