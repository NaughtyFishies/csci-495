import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from data.vocabulary import vocabulary
import re
from transformer import eval_interval, eval_iters, get_batch, GPTLanguageModel, device, \
    vocab_size, learning_rate, max_iters, words, stoi, itos, block_size, n_embd, n_head, n_layer, dropout

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    model = GPTLanguageModel()
    m = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print(f"Vocabulary size: {vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate 10,000 tokens
    # context = torch.tensor([[stoi["<sos>"]]], dtype=torch.long, device=device)
    # generated = m.generate(context, max_new_tokens=10000)[0].tolist()
    # generated_text = ' '.join([itos[i] for i in generated])

    # print("\nFirst 100 tokens of generated text:")
    # print(' '.join([itos[i] for i in generated[:100]]))

    # output_file = 'generated.txt'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(generated_text)

    # print(f"\nFull output (10,000 tokens) written to {output_file}")

    model_save_path = f'models/{max_iters}_epoch_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': words,
        'stoi': stoi,
        'itos': itos,
        'config': {
            'block_size': block_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
        }
    }, model_save_path)

    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
