import os
import json
import re
from extract_audio import extract_audio
from transcribe_audio import transcribe_audio
from pre_process import preprocess_text
from information_extraction import extract_information
from sentiment_analyzer import analyze_sentiment, get_intensity_summary
from summarization import generate_structured_summary
from insights import generate_all_insights
from audio_volume import get_volume_for_segments, detect_volume_peaks, get_audio_stats


def get_match_name(video_path):
    """Extract clean match name from video filename."""
    filename = os.path.basename(video_path)
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Extract team names pattern: "Team1 X - Y Team2"
    match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+)\s*[-–]\s*(\d+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', name)
    if match:
        team1, score1, score2, team2 = match.groups()
        return f"{team1.replace(' ', '_')}_vs_{team2.replace(' ', '_')}_{score1}-{score2}"
    # Fallback: clean filename
    return re.sub(r'[^\w\s-]', '', name).replace(' ', '_')[:50]


def save_to_file(content, filepath):
    """Save content to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))
    print(f"Saved: {filepath}")


def analyze_match(video_path):
    """Analyze a football match video."""
    # Get match name for output folder
    match_name = get_match_name(video_path)
    output_dir = f"/home/batuhan/vscodeprojects/nlpfootbal/outputs/{match_name}"
    
    print(f"\n{'='*50}")
    print(f"Analyzing: {match_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio
    audio = extract_audio(video_path)
    print(f"Audio extracted: {audio}")
    
    # Transcribe
    result = transcribe_audio(audio)
    text = result["text"]
    segments = result["segments"]
    
    # Save transcript
    save_to_file(text, f"{output_dir}/{match_name}_transcript.txt")
    
    # Preprocess
    prep = preprocess_text(text)
    preprocessed_output = f"Sentences: {len(prep['sentences'])}\nClean Tokens: {len(prep['clean_tokens'])}\n\nClean Text:\n{prep['clean_text']}"
    save_to_file(preprocessed_output, f"{output_dir}/{match_name}_preprocessed.txt")
    
    # Information Extraction
    info = extract_information(text, segments)
    save_to_file(info['events'], f"{output_dir}/{match_name}_events.json")
    
    # Audio Volume Analysis
    print("Analyzing audio volume levels...")
    segments_with_volume = get_volume_for_segments(audio, segments)
    volume_peaks = detect_volume_peaks(audio)
    audio_stats = get_audio_stats(audio)
    
    # Sentiment Analysis (with audio volume)
    analyzed = analyze_sentiment(segments, segments_with_volume)
    
    # Generate Summary
    video_title = os.path.basename(video_path)
    match_summary = generate_structured_summary(text, info['events'], info['entities'], video_title)
    save_to_file(match_summary, f"{output_dir}/{match_name}_summary.txt")
    
    # Generate Insights
    insights = generate_all_insights(
        text, info['events'], info['entities'], 
        analyzed, prep['clean_tokens'], output_dir
    )
    
    # Rename insight files with match name
    for old_name in ['excitement_graph.png', 'event_timeline.png', 'match_insights.json', 'match_events.csv']:
        old_path = f"{output_dir}/{old_name}"
        new_path = f"{output_dir}/{match_name}_{old_name}"
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
    
    print(f"\n=== Analysis Complete ===")
    print(f"All outputs saved to: {output_dir}/")
    return output_dir


if __name__ == "__main__":
    video = "/home/batuhan/vscodeprojects/nlpfootbal/FULL MATCH Portugal v Spain  2018 FIFA World Cup.mp4"
    analyze_match(video)
