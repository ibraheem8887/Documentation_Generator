#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================================================
# Phase 3 - Word2Vec Evaluation Notebook
# =====================================================

import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from bpe.bpe_tokenizer import BPETokenizer


# In[1]:


# ----------------------------
# 1. Load vocab + model
# ----------------------------
# Load vocab (rebuild the same way as training)
df = pd.read_csv("data/clean_dataset.csv")

# Load BPE models
with open("bpe/bpe_code.pkl", "rb") as f:
    bpe_code = pickle.load(f)

with open("bpe/bpe_doc.pkl", "rb") as f:
    bpe_doc = pickle.load(f)

# For evaluation, rebuild a small vocab from sample
sample_df = df.sample(200, random_state=42)
all_tokens = [bpe_code.encode(code) for code in sample_df['code'].astype(str)] + \
             [bpe_doc.encode(doc) for doc in sample_df['docstring'].astype(str)]

flat_tokens = [tok for seq in all_tokens for tok in seq]
vocab = list(set(flat_tokens))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print("Vocab size:", len(vocab))

# Define Word2Vec (must match training!)
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        emb = self.embeddings(x)
        return self.output(emb)

# Load trained model
EMBED_DIM = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Word2Vec(len(vocab), EMBED_DIM).to(device)
model.load_state_dict(torch.load("word2vec/word2vec_model.pt", map_location=device))
model.eval()

print("Model loaded ✅")


# In[ ]:


import torch.nn.functional as F

def get_vector(word):
    idx = word_to_idx.get(word, None)
    if idx is None:
        return None
    tensor = torch.tensor([idx], dtype=torch.long).to(device)
    return model.embeddings(tensor).detach().cpu().numpy()[0]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(word, top_n=5):
    vec = get_vector(word)
    if vec is None:
        return []
    sims = {}
    for other in vocab:
        if other == word: continue
        o_vec = get_vector(other)
        if o_vec is not None:
            sims[other] = cosine_similarity(vec, o_vec)
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]


# In[ ]:


test_words = ["def", "class", "import", "return", "self"]  # pick words likely in vocab
for w in test_words:
    print(f"\nMost similar to '{w}':")
    for sim_word, score in most_similar(w, top_n=5):
        print(f"  {sim_word} -> {score:.3f}")


# In[ ]:


# Get vectors for a random sample of vocab
sample_words = np.random.choice(vocab, size=50, replace=False)
vectors = np.array([get_vector(w) for w in sample_words])

# Reduce dimensions
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10,8))
plt.scatter(reduced[:,0], reduced[:,1], c="blue")

for i, word in enumerate(sample_words):
    plt.annotate(word, (reduced[i,0], reduced[i,1]))

plt.title("Word2Vec Embedding Visualization (PCA)")
plt.show()

