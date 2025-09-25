"""
Unified Pipeline for Documentation Generation System

This module integrates BPE tokenization, Word2Vec embeddings, and BiLSTM language model
to generate docstrings for Python functions.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import ast
import re
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bpe.bpe_tokenizer import BPETokenizer
from word2vec.word2vec import Word2Vec


class BiLSTMLM(nn.Module):
    """BiLSTM Language Model for text generation"""
    
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class DocumentationPipeline:
    """
    Unified pipeline for generating documentation from Python code.
    
    Pipeline steps:
    1. Parse Python function code
    2. BPE tokenization
    3. Word2Vec embeddings (optional enhancement)
    4. BiLSTM language model generates docstring
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the documentation pipeline.
        
        Args:
            model_dir: Directory containing trained models. Defaults to parent directories.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or os.path.dirname(os.path.dirname(__file__))
        
        # Initialize components
        self.tokenizer = None
        self.word2vec_model = None
        self.language_model = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Load BPE tokenizer
            tokenizer_path = os.path.join(self.model_dir, 'bilstm', 'bpe_tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                tok_data = pickle.load(f)
            
            if isinstance(tok_data, dict):
                self.tokenizer = BPETokenizer()
                self.tokenizer.vocab = tok_data.get("vocab", [])
                self.tokenizer.merges = tok_data.get("merges", [])
                # Use pre-computed mappings if available, otherwise create them
                if "word_to_idx" in tok_data and "idx_to_word" in tok_data:
                    self.tokenizer.word_to_idx = tok_data["word_to_idx"]
                    self.tokenizer.idx_to_word = tok_data["idx_to_word"]
                else:
                    # Create word mappings from vocab list
                    vocab_list = self.tokenizer.vocab if isinstance(self.tokenizer.vocab, list) else []
                    self.tokenizer.word_to_idx = {w: i for i, w in enumerate(vocab_list)}
                    self.tokenizer.idx_to_word = {i: w for w, i in self.tokenizer.word_to_idx.items()}
            else:
                self.tokenizer = tok_data
            
            vocab_size = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') and self.tokenizer.vocab else 0
            print(f"✅ BPE Tokenizer loaded | Vocab size: {vocab_size}")
            
            # Load Word2Vec model (optional)
            try:
                word2vec_path = os.path.join(self.model_dir, 'word2vec', 'word2vec_model.pt')
                if os.path.exists(word2vec_path):
                    w2v_checkpoint = torch.load(word2vec_path, map_location=self.device)
                    vocab_size = w2v_checkpoint.get('vocab_size', 1000)
                    embed_dim = w2v_checkpoint.get('embed_dim', 100)
                    
                    self.word2vec_model = Word2Vec(vocab_size, embed_dim)
                    self.word2vec_model.load_state_dict(w2v_checkpoint['model_state_dict'])
                    self.word2vec_model.to(self.device)
                    print("✅ Word2Vec model loaded")
            except Exception as e:
                print(f"⚠️ Word2Vec model not loaded: {e}")
            
            # Load BiLSTM Language Model
            bilstm_path = os.path.join(self.model_dir, 'bilstm', 'bilstm_model.pt')
            checkpoint = torch.load(bilstm_path, map_location=self.device)
            
            config = checkpoint.get("model_config", {})
            self.language_model = BiLSTMLM(
                vocab_size=checkpoint["vocab_size"],
                embed_dim=config.get("embed_dim", 100),
                hidden_dim=config.get("hidden_dim", 128),
                num_layers=config.get("num_layers", 2)
            ).to(self.device)
            
            self.language_model.load_state_dict(checkpoint["model_state_dict"])
            self.language_model.eval()
            print("✅ BiLSTM Language Model loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def extract_function_info(self, code: str) -> Dict:
        """
        Extract function information from Python code.
        
        Args:
            code: Python function code as string
            
        Returns:
            Dictionary containing function name, parameters, body, etc.
        """
        try:
            tree = ast.parse(code)
            
            # Find function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break
            
            if not func_def:
                raise ValueError("No function definition found in code")
            
            # Extract function information
            func_info = {
                'name': func_def.name,
                'args': [arg.arg for arg in func_def.args.args],
                'body': ast.get_source_segment(code, func_def.body[0]) if func_def.body else "",
                'returns': ast.get_source_segment(code, func_def.returns) if func_def.returns else None,
                'raw_code': code
            }
            
            return func_info
            
        except Exception as e:
            print(f"Error parsing function: {e}")
            # Fallback: treat entire code as function body
            return {
                'name': 'unknown_function',
                'args': [],
                'body': code,
                'returns': None,
                'raw_code': code
            }
    
    def preprocess_code(self, code: str) -> str:
        """
        Preprocess code for tokenization.
        
        Args:
            code: Raw Python code
            
        Returns:
            Preprocessed code string
        """
        # Remove excessive whitespace
        code = re.sub(r'\n\s*\n', '\n', code)
        # Remove comments (optional)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Normalize whitespace
        code = ' '.join(code.split())
        
        return code
    
    def tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code using BPE tokenizer.
        
        Args:
            code: Preprocessed code string
            
        Returns:
            List of tokens
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        tokens = self.tokenizer.encode(code)
        return tokens
    
    def generate_docstring(self, code: str, max_length: int = 100) -> str:
        """
        Generate docstring using hybrid approach: AI model + templates.

        Args:
            code: Python function code
            max_length: Maximum length of generated docstring

        Returns:
            Generated docstring
        """
        # Try AI generation first
        ai_docstring = self._try_ai_generation(code, max_length)

        # If AI generation fails or produces poor results, use template
        if not ai_docstring or self._is_poor_quality(ai_docstring):
            return self._generate_template_docstring(code)

        return ai_docstring

    def _try_ai_generation(self, code: str, max_length: int) -> str:
        """Try to generate docstring using AI model"""
        try:
            if not self.language_model or not self.tokenizer:
                return ""

            # Preprocess and tokenize
            processed_code = self.preprocess_code(code)
            tokens = self.tokenize_code(processed_code)

            # Convert tokens to indices
            token_indices = []
            for token in tokens[:50]:  # Limit input length
                if hasattr(self.tokenizer, 'word_to_idx') and token in self.tokenizer.word_to_idx:
                    token_indices.append(self.tokenizer.word_to_idx[token])
                else:
                    token_indices.append(0)  # Unknown token

            if not token_indices:
                return ""

            # Generate text using language model
            generated_tokens = self._generate_text(token_indices, max_length)

            # Convert back to text
            docstring = self._tokens_to_docstring(generated_tokens)

            return docstring
        except Exception:
            return ""

    def _is_poor_quality(self, docstring: str) -> bool:
        """Check if generated docstring is of poor quality"""
        if not docstring or len(docstring) < 10:
            return True

        # Check for fallback template
        if "Generated docstring for this function" in docstring:
            return True

        # Check for excessive repetition
        words = docstring.replace('"""', '').strip().split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 3:  # Too repetitive
                return True

        return False

    def _generate_template_docstring(self, code: str) -> str:
        """Generate docstring using templates based on code analysis"""
        func_info = self.extract_function_info(code)
        func_name = func_info.get('name', 'function')
        args = func_info.get('args', [])

        # Analyze function name for hints
        name_lower = func_name.lower()

        # Template selection based on function name patterns
        if 'calculate' in name_lower or 'compute' in name_lower:
            purpose = f"Calculate and return a computed value"
        elif 'get' in name_lower or 'fetch' in name_lower:
            purpose = f"Retrieve and return data"
        elif 'set' in name_lower or 'update' in name_lower:
            purpose = f"Update or modify data"
        elif 'is_' in name_lower or 'check' in name_lower:
            purpose = f"Check a condition and return boolean result"
        elif 'find' in name_lower or 'search' in name_lower:
            purpose = f"Search for and return matching items"
        elif 'create' in name_lower or 'make' in name_lower:
            purpose = f"Create and return a new object"
        elif 'process' in name_lower or 'handle' in name_lower:
            purpose = f"Process input data and return results"
        else:
            purpose = f"Perform the {func_name} operation"

        # Build docstring
        docstring_parts = [purpose + "."]

        if args:
            docstring_parts.append("")
            docstring_parts.append("Args:")
            for arg in args:
                if arg != 'self':
                    docstring_parts.append(f"    {arg}: Input parameter")

        docstring_parts.append("")
        docstring_parts.append("Returns:")
        docstring_parts.append("    The result of the operation")

        docstring_content = "\n".join(docstring_parts)
        return f'"""\n{docstring_content}\n"""'
    
    def _generate_text(self, start_tokens: List[int], max_length: int) -> List[int]:
        """Generate text using the language model with improved sampling"""
        self.language_model.eval()

        tokens = start_tokens.copy()
        generated_tokens = []
        repetition_penalty = {}

        with torch.no_grad():
            for step in range(max_length):
                # Use last 20 tokens as context
                context = tokens[-20:]
                inp = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.device)

                # Get model output
                out = self.language_model(inp)
                logits = out[0, -1]

                # Apply repetition penalty
                for token_id, count in repetition_penalty.items():
                    if count > 2:  # Penalize tokens that appear more than twice
                        logits[token_id] -= 2.0

                # Use temperature sampling instead of argmax
                temperature = 0.8
                logits = logits / temperature

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample from top-k tokens to add variety
                top_k = 50
                top_k_probs, top_k_indices = torch.topk(probs, top_k)

                # Sample from the top-k distribution
                sampled_idx = torch.multinomial(top_k_probs, 1).item()
                next_token = top_k_indices[sampled_idx].item()

                # Update repetition tracking
                repetition_penalty[next_token] = repetition_penalty.get(next_token, 0) + 1

                tokens.append(next_token)
                generated_tokens.append(next_token)

                # Stop conditions
                if next_token >= len(getattr(self.tokenizer, 'idx_to_word', {})):
                    break

                # Stop if we generate end-of-word or punctuation
                if hasattr(self.tokenizer, 'idx_to_word'):
                    token_str = self.tokenizer.idx_to_word.get(next_token, '')
                    if token_str in ['</w>', '.', '!', '?', '\n']:
                        break

        return generated_tokens
    
    def _tokens_to_docstring(self, tokens: List[int]) -> str:
        """Convert token indices back to docstring text with better formatting"""
        if not hasattr(self.tokenizer, 'idx_to_word'):
            return '"""\nGenerated docstring (token conversion not available)\n"""'

        words = []
        for token_idx in tokens:
            if token_idx in self.tokenizer.idx_to_word:
                word = self.tokenizer.idx_to_word[token_idx]
                # Skip very repetitive or meaningless tokens
                if word not in ['path', '*', '=', '_', '(', ')', ','] or len(words) < 5:
                    words.append(word)

        # Join and clean up
        text = ''.join(words)
        text = text.replace('</w>', ' ')  # BPE end-of-word marker
        text = ' '.join(text.split())  # Normalize whitespace

        # Remove excessive repetition
        words_list = text.split()
        if len(words_list) > 3:
            # Remove consecutive duplicate words
            cleaned_words = [words_list[0]]
            for word in words_list[1:]:
                if word != cleaned_words[-1]:
                    cleaned_words.append(word)
            text = ' '.join(cleaned_words)

        # Fallback to template if text is too repetitive or empty
        if not text or len(set(text.split())) < 3:
            return '"""\nGenerated docstring for this function.\n"""'

        # Format as docstring
        return f'"""\n{text.strip()}\n"""'
    
    def generate_documentation(self, code: str) -> Dict:
        """
        Generate complete documentation for a Python function.
        
        Args:
            code: Python function code
            
        Returns:
            Dictionary containing function info, summary, and docstring
        """
        # Extract function information
        func_info = self.extract_function_info(code)
        
        # Generate docstring
        docstring = self.generate_docstring(code)
        
        # Create summary
        summary = self._create_summary(func_info)
        
        return {
            'function_info': func_info,
            'summary': summary,
            'docstring': docstring,
            'full_documentation': self._format_full_documentation(func_info, summary, docstring)
        }
    
    def _create_summary(self, func_info: Dict) -> str:
        """Create a summary of the function"""
        name = func_info.get('name', 'unknown')
        args = func_info.get('args', [])
        
        if args:
            args_str = ', '.join(args)
            summary = f"Function '{name}' takes parameters: {args_str}"
        else:
            summary = f"Function '{name}' takes no parameters"
        
        return summary
    
    def _format_full_documentation(self, func_info: Dict, summary: str, docstring: str) -> str:
        """Format complete documentation"""
        name = func_info.get('name', 'unknown')
        args = func_info.get('args', [])
        
        doc = f"# Function: {name}\n\n"
        doc += f"## Summary\n{summary}\n\n"
        doc += f"## Parameters\n"
        
        if args:
            for arg in args:
                doc += f"- `{arg}`: Parameter description\n"
        else:
            doc += "No parameters\n"
        
        doc += f"\n## Generated Docstring\n{docstring}\n"
        
        return doc
