from transformers import pipeline
from typing import List, Dict
import re


# Load BART model for summarization
summarizer = None


def extract_teams_and_score_from_title(title: str) -> dict:
    """Extract team names and score from video title."""
    # Pattern 1: "Team1 X - Y Team2" (e.g., "Man City 3 - 0 West Ham")
    match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+)\s*[-–]\s*(\d+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', title)
    if match:
        return {
            "teams": [match.group(1).strip(), match.group(4).strip()],
            "score": f"{match.group(2)}-{match.group(3)}"
        }
    
    # Pattern 2: "Team1 v Team2" or "Team1 vs Team2" (e.g., "Portugal v Spain")
    match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:v|vs)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', title)
    if match:
        return {"teams": [match.group(1).strip(), match.group(2).strip()], "score": None}
    
    return {"teams": [], "score": None}


def load_summarizer():
    """Lazy load the summarization model."""
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer


def generate_highlight_summary(text: str, events: List[Dict] = None, max_length: int = 150) -> str:
    """Generate abstractive summary of the match using BART."""
    model = load_summarizer()
    
    # Truncate text if too long (BART has 1024 token limit)
    if len(text) > 3000:
        text = text[:3000]
    
    # Generate summary
    summary = model(text, max_length=max_length, min_length=50, do_sample=False)
    
    return summary[0]["summary_text"]


def generate_structured_summary(text: str, events: List[Dict], entities: Dict, video_title: str = None) -> str:
    """Generate a structured match summary with events, entities, officials, stadium, and lineup."""
    
    # Extract teams and score from video title
    teams = []
    score = None
    if video_title:
        result = extract_teams_and_score_from_title(video_title)
        teams = result["teams"]
        score = result["score"]
    
    # Fallback to NER-detected teams
    if not teams:
        teams = entities.get("teams", [])
    
    # Extract additional information
    players = entities.get("persons", [])
    officials = entities.get("officials", [])
    stadiums = entities.get("stadiums", [])
    locations = entities.get("locations", [])
    player_team_map = entities.get("player_team_map", {})
    goals = [e for e in events if e.get("type") == "goal"]
    cards = [e for e in events if e.get("type") in ["yellow_card", "red_card"]]
    substitutions = [e for e in events if e.get("type") == "substitution"]
    
    # Build structured summary
    lines = []
    
    # Match Header with Score
    if teams:
        match_line = f"⚽ Match: {teams[0]} vs {teams[1]}" if len(teams) >= 2 else f"⚽ Match: {teams[0]}"
        if score:
            match_line += f" | Score: {score}"
        lines.append(match_line)
    elif score:
        lines.append(f"⚽ Final Score: {score}")
    
    lines.append("=" * 50)
    
    # Stadium/Venue
    if stadiums:
        lines.append(f"\n🏟️ Stadium: {', '.join(stadiums)}")
    elif locations:
        lines.append(f"\n📍 Location: {', '.join(locations[:2])}")
    
    # Officials/Referees
    if officials:
        lines.append(f"\n👨‍⚖️ Officials: {', '.join(officials)}")
    
    # Filter out non-team entities (leagues, competitions)
    non_team_keywords = ["league", "cup", "premier", "champions", "europa", "fa ", "efl", "carabao"]
    
    def is_valid_team(name):
        return not any(kw in name.lower() for kw in non_team_keywords)
    
    # Use video title teams as primary, filter NER detected teams
    primary_teams = teams[:2] if teams else []
    
    # Group players by team
    team_players = {team: [] for team in primary_teams}
    unassigned_players = []
    
    for player in players:
        assigned_team = player_team_map.get(player)
        
        # Check if assigned team matches any primary team (fuzzy match)
        if assigned_team:
            matched = False
            for pt in primary_teams:
                if (pt.lower() in assigned_team.lower() or 
                    assigned_team.lower() in pt.lower() or
                    any(word in assigned_team.lower() for word in pt.lower().split())):
                    team_players[pt].append(player)
                    matched = True
                    break
            if not matched and is_valid_team(assigned_team):
                team_players.setdefault(assigned_team, []).append(player)
            elif not matched:
                unassigned_players.append(player)
        else:
            unassigned_players.append(player)
    
    # Lineup/Squad by Team
    has_team_players = any(roster for roster in team_players.values())
    if has_team_players or unassigned_players:
        lines.append(f"\n👥 Squad/Lineup:")
        
        # Show primary teams first, then others
        shown_teams = set()
        for team_name in primary_teams:
            if team_name in team_players and team_players[team_name]:
                lines.append(f"\n  📋 {team_name}:")
                for i, player in enumerate(team_players[team_name][:8], 1):
                    lines.append(f"    {i}. {player}")
                if len(team_players[team_name]) > 8:
                    lines.append(f"    ... and {len(team_players[team_name]) - 8} more")
                shown_teams.add(team_name)
        
        # NOTE: Only showing primary teams, skipping other detected teams (e.g., nationalities)
        
        # Show unassigned players if any
        if unassigned_players:
            lines.append(f"\n  📋 Other Players Mentioned:")
            for i, player in enumerate(unassigned_players[:6], 1):
                lines.append(f"    {i}. {player}")
            if len(unassigned_players) > 6:
                lines.append(f"    ... and {len(unassigned_players) - 6} more")
    
    # Goals
    if goals:
        lines.append(f"\n⚽ Goals ({len(goals)}):")
        for i, goal in enumerate(goals, 1):
            time_str = f"{int(goal.get('time', 0) // 60)}'" if 'time' in goal else ""
            lines.append(f"  {i}. [{time_str}] {goal.get('text', '')[:80]}...")
    
    # Cards
    if cards:
        lines.append(f"\n🟨🟥 Cards ({len(cards)}):")
        for card in cards:
            card_emoji = "🟨" if card.get("type") == "yellow_card" else "🟥"
            time_str = f"{int(card.get('time', 0) // 60)}'" if 'time' in card else ""
            lines.append(f"  {card_emoji} [{time_str}] {card.get('text', '')[:60]}...")
    
    # Substitutions
    if substitutions:
        lines.append(f"\n🔄 Substitutions ({len(substitutions)}):")
        for sub in substitutions[:5]:
            time_str = f"{int(sub.get('time', 0) // 60)}'" if 'time' in sub else ""
            lines.append(f"  [{time_str}] {sub.get('text', '')[:60]}...")
    
    # Abstractive summary
    if text:
        try:
            abstract = generate_highlight_summary(text)
            lines.append(f"\n📝 Match Summary:\n{abstract}")
        except Exception as e:
            lines.append(f"\n(Summary generation failed: {e})")
    
    return "\n".join(lines)
