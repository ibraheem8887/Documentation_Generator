import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, center_words):
        embeds = self.embeddings(center_words)
        out = self.output(embeds)
        return out
    
    def get_embedding(self, word_idx):
        return self.embeddings.weight[word_idx]
