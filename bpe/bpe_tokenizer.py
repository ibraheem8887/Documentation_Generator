import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []

    def get_stats(self, corpus):
        """Count frequency of symbol pairs"""
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        """Merge the most frequent pair"""
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_corpus = {}
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def train(self, texts):
        """Train BPE on a list of strings"""
        # Initialize vocab as char-level tokens
        corpus = defaultdict(int)
        for text in texts:
            word = ' '.join(list(text)) + ' </w>'
            corpus[word] += 1

        for i in range(self.num_merges):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.merges.append(best)

        self.vocab = corpus

    def encode(self, text):
        """Encode text using learned merges"""
        word = list(text) + ["</w>"]
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            if not pairs:
                break
            pair_freq = {p: self.merges.index(p) if p in self.merges else float('inf') for p in pairs}
            best = min(pair_freq, key=pair_freq.get)
            if pair_freq[best] == float('inf'):
                break
            i = pairs.index(best)
            word[i:i+2] = [''.join(best)]
        return word

    def decode(self, tokens):
        """Decode back into string"""
        return ''.join(tokens).replace("</w>", "")
