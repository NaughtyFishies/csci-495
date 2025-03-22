import json
import re

vocabulary = [
    # pronouns
    "i", "you", "they", "we",
    # verbs 
    "move", "take", "see", "say", "know", "want", "touch", "love", "hate", "think", "use", "are", "fly", "sleep", "stand", 
    # graph
    "right", "left", "north", "south", "east", "west",
    # grammatical
    "no", "ha", "yes", "ed", "will", "ing",
    # Conjunctions
    "and", "or",
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "many",
    # Prepositions
    "in", "on", "at", "with",
    # Nouns
    "food", "water", "house", "person", "child", "tree", "sun", "moon", "sky", "bird", "dog", "cat", "road", "day", "night",
    # Adjectives
    "big", "small", "good", "bad", "hot", "cold", "red", "blue", "yellow", "green", "happy", "sad"
]

with open('response.json', 'r') as file:
    story_list = json.load(file)

def preprocess_string(s):
    s = re.sub(r'\*\*.*?\*\*', '', s)
    s = s.lower()
    s = s.replace('\n', '')
    s = s.replace('\"', '')
    return s.strip()

processed_string_list = [preprocess_string(s) for s in story_list]

def validate_string(s, allowed_words):
    s = re.sub(r'\*\*.*?\*\*', '', s)
    s = s.lower()
    s = s.replace('\n', '')
    s = s.replace('\\"', '')
    words = re.findall(r'\b\w+\b', s)
    invalid_words = [word for word in words if word not in allowed_words]
    return invalid_words

for i, s in enumerate(processed_string_list):
    invalid_words = validate_string(s, vocabulary)
    if invalid_words:
        print(f"String {i} contains invalid words: {invalid_words}")
    else:
        print(f"String {i} is valid.")

valid_strings = [s for s in processed_string_list if not validate_string(s, vocabulary)]

with open('valid_strings.json', 'w') as file:
    json.dump(valid_strings, file)