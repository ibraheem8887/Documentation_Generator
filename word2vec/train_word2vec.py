import torch
import torch.nn as nn
import torch.optim as optim
import random
from word2vec.word2vec import Word2Vec

def train_word2vec(pairs, vocab, embed_dim=100, epochs=5, lr=0.01, num_neg=5):
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    # Convert pairs to indices
    data = [(word_to_idx[c], word_to_idx[o]) for c, o in pairs if c in word_to_idx and o in word_to_idx]
    
    model = Word2Vec(len(vocab), embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vocab_size = len(vocab)
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)  # Shuffle for better training
        for center, context in data:
            # Sample negatives
            negs = []
            for _ in range(num_neg):
                neg = random.randint(0, vocab_size - 1)
                while neg == context:
                    neg = random.randint(0, vocab_size - 1)
                negs.append(neg)
            
            center_tensor = torch.tensor([center], dtype=torch.long)
            pos_tensor = torch.tensor([context], dtype=torch.long)
            neg_tensors = torch.tensor(negs, dtype=torch.long)
            
            # Get logits
            logits = model(center_tensor).squeeze()
            pos_logit = logits[pos_tensor]
            neg_logits = logits[neg_tensors]
            
            # Loss: maximize log sigmoid(pos) + sum log sigmoid(-neg)
            pos_loss = -torch.log(torch.sigmoid(pos_logit))
            neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_logits)))
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    return model, word_to_idx, idx_to_word
