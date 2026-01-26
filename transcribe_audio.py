import whisper

def transcribe_audio(audio_path: str, model_name: str = "base") -> dict:
    """Transcribe audio to text with timestamps using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    return {
        "text": result["text"],
        "segments": segments
    }