import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


def load_audio(audio_path: str) -> Tuple[int, np.ndarray]:
    """Load audio file and return sample rate and data."""
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert stereo to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize to float
    audio_data = audio_data.astype(np.float32)
    if audio_data.max() > 1:
        audio_data = audio_data / np.abs(audio_data).max()
    
    return sample_rate, audio_data


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))


def calculate_peak(audio_chunk: np.ndarray) -> float:
    """Calculate peak amplitude of audio chunk."""
    return np.max(np.abs(audio_chunk))


def get_volume_for_segments(audio_path: str, segments: List[Dict]) -> List[Dict]:
    """Calculate volume metrics for each transcript segment."""
    sample_rate, audio_data = load_audio(audio_path)
    total_duration = len(audio_data) / sample_rate
    
    # Calculate global stats for normalization
    global_rms = calculate_rms(audio_data)
    global_peak = calculate_peak(audio_data)
    
    results = []
    for segment in segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", start_time + 5)
        
        # Convert time to sample indices
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Clamp to valid range
        start_idx = max(0, min(start_idx, len(audio_data) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(audio_data)))
        
        # Extract audio chunk for this segment
        chunk = audio_data[start_idx:end_idx]
        
        if len(chunk) > 0:
            rms = calculate_rms(chunk)
            peak = calculate_peak(chunk)
            
            # Normalize to 0-1 range relative to global stats
            rms_normalized = min(rms / (global_rms * 2), 1.0) if global_rms > 0 else 0
            peak_normalized = min(peak / global_peak, 1.0) if global_peak > 0 else 0
            
            # Combined volume score
            volume_score = (rms_normalized * 0.7 + peak_normalized * 0.3)
        else:
            rms_normalized = 0
            peak_normalized = 0
            volume_score = 0
        
        results.append({
            **segment,
            "rms": float(round(rms_normalized, 3)),
            "peak": float(round(peak_normalized, 3)),
            "volume_score": float(round(volume_score, 3))
        })
    
    return results


def get_volume_timeline(audio_path: str, window_seconds: float = 2.0) -> List[Dict]:
    """Get volume levels over time with fixed time windows."""
    sample_rate, audio_data = load_audio(audio_path)
    total_duration = len(audio_data) / sample_rate
    
    window_samples = int(window_seconds * sample_rate)
    global_rms = calculate_rms(audio_data)
    
    timeline = []
    for i in range(0, len(audio_data), window_samples):
        chunk = audio_data[i:i + window_samples]
        if len(chunk) > 0:
            rms = calculate_rms(chunk)
            rms_normalized = min(rms / (global_rms * 2), 1.0) if global_rms > 0 else 0
            
            timeline.append({
                "time": float(round(i / sample_rate, 2)),
                "volume": float(round(rms_normalized, 3))
            })
    
    return timeline


def detect_volume_peaks(audio_path: str, threshold: float = 0.6, min_gap_seconds: float = 10.0) -> List[Dict]:
    """Detect moments where volume exceeds threshold (likely exciting moments)."""
    timeline = get_volume_timeline(audio_path, window_seconds=1.0)
    
    peaks = []
    last_peak_time = -min_gap_seconds
    
    for point in timeline:
        if point["volume"] >= threshold:
            if point["time"] - last_peak_time >= min_gap_seconds:
                peaks.append({
                    "time": float(point["time"]),
                    "volume": float(point["volume"]),
                    "type": "high_volume_moment"
                })
                last_peak_time = point["time"]
    
    return peaks


def get_audio_stats(audio_path: str) -> Dict:
    """Get overall audio statistics."""
    sample_rate, audio_data = load_audio(audio_path)
    
    return {
        "duration_seconds": float(round(len(audio_data) / sample_rate, 2)),
        "sample_rate": int(sample_rate),
        "average_volume": float(round(float(calculate_rms(audio_data)), 4)),
        "peak_volume": float(round(float(calculate_peak(audio_data)), 4))
    }
