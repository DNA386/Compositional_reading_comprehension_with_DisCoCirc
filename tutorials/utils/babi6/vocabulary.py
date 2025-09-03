# Nouns
PEOPLE = [
    "Andrew", "Bill", "Clara", "Denise", "Eric", "Fred", "Gillian", "Heidi",
    # Original Babi
    "Mary", "Sandra", "Daniel", "john",
]  # 8
people = [p.lower() for p in PEOPLE]

OBJECTS = ["apple", "football", "milk", "slippers"]  # 4
LOCATIONS = ["kitchen", "office", "hallway", "bedroom", "garden", "bathroom", "cinema", "park"]  # 8
NOUNS = set(PEOPLE + people + OBJECTS + LOCATIONS)


# Verbs
MOVE = ["journeyed", "moved", "moved back", "travelled", "went", "went back"]  # 6
GRAB = ["grabbed", "picked up", "took", "got"]  # 4
DROP = ["discarded", "dropped", "left", "put down"]  # 4

VERB_PHRASES = GRAB + DROP + [f"{move} to" for move in MOVE]

ITV = ["journeyed", "moved", "travelled", "went"]
TV = [
    "discarded", "dropped", "left", "put",
    "grabbed", "picked", "took", "got"
]
PREP = ["up", "down"]
