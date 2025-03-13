import torch
import torch.nn as nn

vocab = {
    "I": 1, "you": 2, "he": 3, "she": 4, "we": 5, "they": 6,
    "go": 7, "come": 8, "eat": 9, "drink": 10, "see": 11, "speak": 12, "know": 13, "want": 14, "run": 15, "love": 16, "think": 17,
    "right": 18, "no right": 19, "ing": 20, "no ing": 21, "north": 22, "south": 23, "east": 24, "west": 25,
    "ed": 26, "will": 27, "no": 28, "ha": 29
}

vocab_size = len(vocab) + 1  # +1 for padding/unknown tokens
embedding_dim = 100 # Tweek depending on what we choose
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

sentence = "I will run right"
token_ids = [vocab[word] for word in sentence.split()]

input_tensor = torch.tensor([token_ids])
embeddings = embedding_layer(input_tensor)

print("Token IDs:", token_ids)
print("Embeddings:", embeddings)