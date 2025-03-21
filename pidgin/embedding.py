import torch
import torch.nn as nn


def create_numbered_vocab(word_list):
    return {word: idx + 1 for idx, word in enumerate(word_list)}

# Example usage
word_list = [
    # No gender
    # no capitalization

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
    "food", "water", "house", "person", "child", "tree", "sun", "moon", "sky", "bird", "dog", "cat", "road", 
    # Adjectives
    "big", "small", "good", "bad", "hot", "cold", "red", "blue", "yellow", "green"
]

vocab = create_numbered_vocab(word_list)

vocab_size = len(vocab) + 1  # +1 for padding/unknown tokens
embedding_dim = 100 # Tweek depending on what we choose
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

sentence = "I will run right"
token_ids = [vocab[word] for word in sentence.split()]

input_tensor = torch.tensor([token_ids])
embeddings = embedding_layer(input_tensor)

print("Token IDs:", token_ids)
print("Embeddings:", embeddings)