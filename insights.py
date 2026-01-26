import json
import csv
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import os


def generate_event_timeline(events: List[Dict]) -> List[Dict]:
    """Generate chronological timeline of match events."""
    timeline = []
    for event in sorted(events, key=lambda x: x.get("time", 0)):
        timeline.append({
            "time": round(event.get("time", 0), 1),
            "minute": int(event.get("time", 0) // 60),
            "type": event.get("type", "unknown"),
            "description": event.get("text", "")[:100]
        })
    return timeline


def calculate_player_mentions(text: str, players: List[str]) -> Dict[str, int]:
    """Count player mentions as performance indicator."""
    text_lower = text.lower()
    mentions = {}
    for player in players:
        count = text_lower.count(player.lower())
        if count > 0:
            mentions[player] = count
    return dict(sorted(mentions.items(), key=lambda x: x[1], reverse=True))


def calculate_team_momentum(segments: List[Dict], team_keywords: Dict[str, List[str]]) -> List[Dict]:
    """Calculate team momentum over time based on mentions."""
    momentum = []
    for seg in segments:
        text_lower = seg.get("text", "").lower()
        time = seg.get("start", 0)
        
        scores = {}
        for team, keywords in team_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[team] = score
        
        momentum.append({"time": time, **scores})
    return momentum


def generate_excitement_data(analyzed_segments: List[Dict]) -> List[Dict]:
    """Generate excitement graph data from sentiment analysis."""
    return [
        {"time": seg["time"], "intensity": seg["intensity"], "mood": seg["mood"]}
        for seg in analyzed_segments
    ]


def extract_keywords(clean_tokens: List[str], top_n: int = 20) -> List[tuple]:
    """Extract top keywords from clean tokens."""
    counter = Counter(clean_tokens)
    return counter.most_common(top_n)


def generate_topic_clusters(clean_tokens: List[str]) -> Dict[str, List[str]]:
    """Cluster tokens into football-related topics."""
    clusters = {
        "players": [],
        "actions": [],
        "positions": [],
        "events": [],
        "other": []
    }
    
    action_words = {"pass", "shot", "goal", "save", "tackle", "cross", "header", "kick"}
    position_words = {"box", "penalty", "corner", "halfway", "goal", "line"}
    event_words = {"goal", "foul", "card", "offside", "substitution", "injury"}
    
    for token in set(clean_tokens):
        if token in action_words:
            clusters["actions"].append(token)
        elif token in position_words:
            clusters["positions"].append(token)
        elif token in event_words:
            clusters["events"].append(token)
        elif token[0].isupper() if token else False:
            clusters["players"].append(token)
        else:
            clusters["other"].append(token)
    
    return {k: v[:10] for k, v in clusters.items()}


def plot_excitement_graph(excitement_data: List[Dict], output_path: str = "excitement_graph.png"):
    """Plot excitement intensity over time."""
    times = [d["time"] / 60 for d in excitement_data]  # Convert to minutes
    intensities = [d["intensity"] for d in excitement_data]
    
    plt.figure(figsize=(12, 4))
    plt.plot(times, intensities, color='blue', alpha=0.7)
    plt.fill_between(times, intensities, alpha=0.3)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Excitement Intensity")
    plt.title("Commentary Excitement Over Time")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_event_timeline(events: List[Dict], output_path: str = "event_timeline.png"):
    """Plot event timeline."""
    if not events:
        return None
    
    event_types = list(set(e["type"] for e in events))
    colors = {"goal": "green", "foul": "orange", "yellow_card": "yellow", 
              "red_card": "red", "substitution": "blue", "injury": "purple"}
    
    plt.figure(figsize=(12, 3))
    for event in events:
        time = event.get("time", 0) / 60
        etype = event["type"]
        color = colors.get(etype, "gray")
        plt.axvline(x=time, color=color, alpha=0.7, linewidth=2, label=etype)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.xlabel("Time (minutes)")
    plt.title("Match Event Timeline")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def export_to_json(data: Dict, output_path: str = "match_insights.json"):
    """Export insights to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    return output_path


def export_to_csv(events: List[Dict], output_path: str = "match_events.csv"):
    """Export events to CSV file."""
    if not events:
        return None
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["time", "minute", "type", "description"])
        writer.writeheader()
        for event in events:
            writer.writerow({
                "time": event.get("time", 0),
                "minute": int(event.get("time", 0) // 60),
                "type": event.get("type", ""),
                "description": event.get("text", "")[:100]
            })
    return output_path


def generate_all_insights(text: str, events: List[Dict], entities: Dict, 
                          analyzed_segments: List[Dict], clean_tokens: List[str],
                          output_dir: str = ".") -> Dict:
    """Generate all insights and export files."""
    
    insights = {
        "timeline": generate_event_timeline(events),
        "player_mentions": calculate_player_mentions(text, entities.get("persons", [])),
        "excitement_data": generate_excitement_data(analyzed_segments),
        "keywords": extract_keywords(clean_tokens),
        "topic_clusters": generate_topic_clusters(clean_tokens)
    }
    
    # Generate plots
    excitement_path = os.path.join(output_dir, "excitement_graph.png")
    timeline_path = os.path.join(output_dir, "event_timeline.png")
    
    plot_excitement_graph(insights["excitement_data"], excitement_path)
    plot_event_timeline(events, timeline_path)
    
    # Export data
    json_path = os.path.join(output_dir, "match_insights.json")
    csv_path = os.path.join(output_dir, "match_events.csv")
    
    export_to_json(insights, json_path)
    export_to_csv(events, csv_path)
    
    print(f"Exported: {excitement_path}, {timeline_path}, {json_path}, {csv_path}")
    
    return insights
