import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define Word Embedding Model
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)  # Convert word indices to dense vectors

# Define Skip-Gram Dataset
class SkipGramDataset(Dataset):
    def __init__(self, word_pairs, word_to_idx):
        self.data = [(word_to_idx[w1], word_to_idx[w2]) for w1, w2 in word_pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# Sample Vocabulary
vocab = ["cat", "dog", "meow", "bark", "pet", "animal"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Sample word pairs for training (context, target)
word_pairs = [("cat", "meow"), ("dog", "bark"), ("cat", "pet"), ("dog", "animal")]

# Create dataset and dataloader
dataset = SkipGramDataset(word_pairs, word_to_idx)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model and Training Setup
vocab_size = len(vocab)
embed_dim = 10  # Size of word embeddings
lr = 0.01
epochs = 100

model = WordEmbeddingModel(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for target, context in dataloader:
        optimizer.zero_grad()

        target_embed = model(target)
        scores = torch.matmul(target_embed, model.embedding.weight.T)  # Compute similarity
        loss = loss_fn(scores, context)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Function to Retrieve Word Embeddings
def get_embedding(word):
    idx = word_to_idx[word]
    return model.embedding.weight[idx].detach().numpy()

# Example: Get embedding for "cat"
print("Embedding for 'cat':", get_embedding("cat"))
