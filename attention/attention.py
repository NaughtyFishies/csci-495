import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define Self-Attention Layer 
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # Compute similarity
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Normalize with softmax
        out = torch.matmul(attn_weights, V)  # Weighted sum

        return out, attn_weights

# Define Word Embedding Model with Attention 
class WordEmbeddingWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SelfAttention(embed_dim)

    def forward(self, x):
        x = self.embedding(x)  # Convert word indices to embeddings
        attn_output, attn_weights = self.attention(x)  # Apply self-attention
        return attn_output, attn_weights

# Create Skip-Gram Dataset
class SkipGramDataset(Dataset):
    def __init__(self, word_pairs, word_to_idx):
        self.data = [(word_to_idx[w1], word_to_idx[w2]) for w1, w2 in word_pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# Define vocabulary and training pairs
vocab = ["cat", "dog", "meow", "bark", "pet", "animal"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
word_pairs = [("cat", "meow"), ("dog", "bark"), ("cat", "pet"), ("dog", "animal")]

# Create dataset and dataloader
dataset = SkipGramDataset(word_pairs, word_to_idx)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training the model
vocab_size = len(vocab)
embed_dim = 10
lr = 0.01
epochs = 100

model = WordEmbeddingWithAttention(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for target, context in dataloader:
        optimizer.zero_grad()

        attn_output, _ = model(target)  # Get attention-enhanced embeddings
        scores = torch.matmul(attn_output, model.embedding.weight.T)  # Compute similarity
        loss = loss_fn(scores, context)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Retrieve Attention-Enhanced Embeddings
def get_embedding(word):
    idx = torch.tensor([word_to_idx[word]])
    attn_embedding, attn_weights = model(idx)
    return attn_embedding.detach().numpy(), attn_weights.detach().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_simularity(embed1, embed2):
    return cosine_similarity(embed1.flatten(), embed2.flatten())
    

# Example: Get embedding for "cat"
cat_embedding, _ = get_embedding("cat")
dog_embedding, _ = get_embedding("dog")
meow_embedding, _ = get_embedding("meow")
print("\nEmbedding for 'cat': ", cat_embedding)
print("Embedding for 'dog': ", dog_embedding)
print("Embedding for 'meow': ", meow_embedding)
print("\nCat-dog similarity: ", find_simularity(cat_embedding, dog_embedding))
print("Cat-meow similarity: ", find_simularity(cat_embedding, meow_embedding))
print("Dog-meow similarity: ", find_simularity(dog_embedding, meow_embedding))

