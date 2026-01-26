import subprocess
import os


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio from video using ffmpeg."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".wav"
    
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-y",
        output_path
    ], check=True, capture_output=True)
    
    return output_path
