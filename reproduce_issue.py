from information_extraction import detect_events_with_timestamps

segments = [
    {"text": "But there is no change in the score line in Sephora.", "start": 64*60},
    {"text": "And still Portugal waits to make the change.", "start": 70*60},
    {"text": "And now the change will be made.", "start": 71*60}, # This might still be caught if context matches, but "change" was the key.
    {"text": "The manager decides to make a substitution.", "start": 72*60}, # Should be caught
    {"text": "Ronaldo comes on for Portugal.", "start": 73*60}, # Should be caught
]

events = detect_events_with_timestamps(segments)
print(f"Detected {len(events)} events.")
for e in events:
    print(f"- {e['type']}: {e['text']}")
