import spacy
import re
from typing import List, Dict

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    nlp = None

GOAL_TIME_WINDOW = 15
EVENT_TIME_WINDOW = 10

FOOTBALL_CONTEXT_PATTERNS = [
    r"\b(goal|score|win|match|game|play)\b",
    r"\b(premier league|cup|final|semifinal)\b",
    r"\b(striker|defender|goalkeeper|midfielder|manager)\b",
    r"\b(kick|shot|corner|penalty|free kick)\b"
]

STADIUM_KEYWORDS = [
    "stadium", "arena", "ground", "park", "field"
]

OFFICIAL_KEYWORDS = [
    "referee", "ref", "official", "linesman", "assistant referee",
    "fourth official", "var", "video assistant", "assistant", "in charge"
]

EVENT_PATTERNS = {
    "goal": [
        r"\b(he scores|scores!|scored!|it'?s a goal|what a goal|brilliant goal)\b",
        r"\b(puts? it in|into the net|back of the net|finds? the net|roof of the net)\b",
        r"\b(first goal|second goal|third goal|opening goal|opens? the scoring)\b",
        r"\b(brilliant finish|lovely finish|clinical finish|tucks it away)\b",
        r"\b(breaks? the deadlock|doubles the lead|seals? the victory)\b",
        r"\b(smashed in the rebound|and it's in)\b",
        r"\b(one null|two null|three null|four null|five null)\b",
        r"\b(one one|two one|three one|two two|three two)\b",
        r"\b(1-0|2-0|3-0|4-0|1-1|2-1|3-1|2-2|3-2)\b"
    ],
    "foul": [
        r"\b(foul|fouls?|fouled|tackle|tackled|brought down|pulled back)\b",
        r"\b(trips?|tripped|push|pushed|shoved)\b"
    ],
    "yellow_card": [
        r"\b(yellow card|booked|booking|caution|cautioned)\b",
        r"\b(shown yellow|gets? a yellow|receives? a yellow)\b"
    ],
    "red_card": [
        r"\b(red card|sent off|sending off|dismissed|ejected)\b",
        r"\b(straight red|second yellow|off he goes)\b"
    ],
    "offside": [
        r"\b(offside|off-?side|flag is up|linesman'?s flag)\b"
    ],
    "substitution": [
        r"\b(substitution|sub|replaced|replacing|goes? off)\b",
        r"\b(brings? on|takes? off)\b"
    ],
    "injury": [
        r"\b(injury|hurt|treatment|stretcher|medical)\b",
        r"\b(holding|injured|limping|medic)\b"
    ]
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    if nlp is None:
        return {"persons": [], "teams": [], "possible_teams": [], "stadiums": [], "locations": [], "officials": [], "player_team_map": {}}
    
    doc = nlp(text)

    teams = set()
    possible_teams = set()
    stadiums = set()
    persons = set()
    locations = set()
    officials = set()
    player_team_map = {}  # Maps player -> team

    for sent in doc.sents:
        sent_lower = sent.text.lower()
        has_football_context = any(
            re.search(p, sent_lower) for p in FOOTBALL_CONTEXT_PATTERNS
        )
        has_official_context = any(
            keyword in sent_lower for keyword in OFFICIAL_KEYWORDS
        )

        # Collect entities in this sentence
        sent_persons = []
        sent_teams = []

        for ent in sent.ents:
            label = ent.label_
            value = ent.text.strip()

            if label == "PERSON":
                if has_official_context:
                    officials.add(value)
                else:
                    persons.add(value)
                    sent_persons.append(value)

            elif label in ["ORG", "NORP", "GPE"]:
                if has_football_context:
                    teams.add(value)
                    sent_teams.append(value)
                else:
                    possible_teams.add(value)

            elif label in ["FAC", "LOC"]:
                if any(k in value.lower() for k in STADIUM_KEYWORDS):
                    stadiums.add(value)
                else:
                    locations.add(value)

        # Associate players with teams mentioned in the same sentence
        if sent_teams and sent_persons:
            primary_team = sent_teams[0]  # Use first team mentioned
            for person in sent_persons:
                if person not in player_team_map:
                    player_team_map[person] = primary_team

    return {
        "persons": list(persons),
        "teams": list(teams),
        "possible_teams": list(possible_teams),
        "stadiums": list(stadiums),
        "locations": list(locations),
        "officials": list(officials),
        "player_team_map": player_team_map
    }


def detect_events_with_timestamps(segments: List[Dict]) -> List[Dict]:
    events = []

    for segment in segments:
        text = segment["text"]
        start_time = segment["start"]
        text_lower = text.lower()

        for event_type, patterns in EVENT_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                events.append({
                    "type": event_type,
                    "text": text.strip(),
                    "time": start_time
                })
                break

    deduplicated = []
    for event in events:
        window = GOAL_TIME_WINDOW if event["type"] == "goal" else EVENT_TIME_WINDOW

        if not any(
            e["type"] == event["type"] and abs(e["time"] - event["time"]) < window
            for e in deduplicated
        ):
            deduplicated.append(event)

    return deduplicated


def extract_information(text: str, segments: List[Dict] = None) -> Dict:
    entities = extract_entities(text)
    events = detect_events_with_timestamps(segments) if segments else []

    summary = {}
    for e in events:
        summary.setdefault(e["type"], []).append(e)

    return {
        "entities": entities,
        "events": events,
        "event_summary": summary
    }