import random

def generate_skipgram_pairs(tokens, window_size=2):
    """
    Generate skip-gram (center, context) pairs from a list of tokens.
    """
    pairs = []
    for i in range(len(tokens)):
        center = tokens[i]
        context_window = tokens[max(0, i - window_size): i] + tokens[i+1: i+window_size+1]
        for context in context_window:
            pairs.append((center, context))
    return pairs
