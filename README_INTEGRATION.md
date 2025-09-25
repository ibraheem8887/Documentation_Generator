# Python Documentation Generation System - Phase 5 Integration

This is the complete integrated system that combines BPE tokenization, Word2Vec embeddings, and BiLSTM language model to automatically generate documentation for Python functions.

## 🏗️ System Architecture

The system follows a unified pipeline:

1. **Input**: Python function code
2. **Step 1**: BPE tokenization for code preprocessing
3. **Step 2**: Word2Vec embeddings for semantic understanding (optional enhancement)
4. **Step 3**: BiLSTM language model generates docstring
5. **Output**: Complete documentation with summary and docstring

## 📁 Project Structure

```
GenerativeAI_Project/
├── integration/
│   ├── pipeline.py           # Core unified pipeline
│   ├── doc_generator.py      # Main documentation generator
│   └── test_integration.py   # Integration tests
├── ui/
│   └── streamlit_app.py      # Web interface
├── bpe/
│   ├── bpe_tokenizer.py      # BPE tokenizer implementation
│   └── bpe_*.pkl             # Trained BPE models
├── word2vec/
│   ├── word2vec.py           # Word2Vec model
│   ├── word2vec_model.pt     # Trained Word2Vec model
│   └── utils.py              # Word2Vec utilities
├── bilstm/
│   ├── bilstm_model.pt       # Trained BiLSTM model
│   └── bpe_tokenizer.pkl     # Tokenizer for BiLSTM
└── data/
    └── clean_dataset.csv     # Training dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Integration

```bash
cd integration
python test_integration.py
```

### 3. Run the Web Interface

```bash
cd ui
streamlit run streamlit_app.py
```

### 4. Command Line Usage

```bash
cd integration
python doc_generator.py "def example(x): return x * 2" -o output.md
```

## 💻 Usage Examples

### Python API

```python
from integration.doc_generator import DocumentationGenerator

# Initialize generator
generator = DocumentationGenerator()

# Generate from code string
code = """
def calculate_area(length, width):
    area = length * width
    return area
"""

result = generator.generate_from_string(code)
print(result['docstring'])

# Generate from file
result = generator.generate_from_file('my_script.py')
print(result['documentation'])
```

### Web Interface

1. Open the Streamlit app in your browser
2. Choose to paste code or upload a Python file
3. Enter your Python function
4. Click "Generate Documentation"
5. Review and download the results

### Command Line

```bash
# Generate from code string
python doc_generator.py -s "def add(a, b): return a + b" -o docs.md

# Generate from file
python doc_generator.py my_script.py -o documentation.md
```

## 🔧 System Components

### 1. DocumentationPipeline (`pipeline.py`)

Core pipeline that integrates all components:
- Loads trained BPE tokenizer, Word2Vec, and BiLSTM models
- Handles code preprocessing and tokenization
- Generates docstrings using the language model
- Provides unified interface for documentation generation

### 2. DocumentationGenerator (`doc_generator.py`)

High-level interface for documentation generation:
- Processes single functions or entire files
- Extracts functions from Python code
- Combines results into formatted documentation
- Handles errors gracefully

### 3. Streamlit UI (`streamlit_app.py`)

Interactive web interface:
- Upload files or paste code
- Real-time documentation generation
- Download generated documentation
- Model status and system information

## 📊 Features

### Input Methods
- ✅ Paste Python code directly
- ✅ Upload Python files (.py)
- ✅ Process multiple functions in a file
- ✅ Command line interface

### Output Formats
- ✅ Generated docstrings
- ✅ Function summaries
- ✅ Complete documentation (Markdown)
- ✅ Downloadable files

### AI Pipeline
- ✅ BPE tokenization for code preprocessing
- ✅ Word2Vec embeddings (optional enhancement)
- ✅ BiLSTM language model for text generation
- ✅ Configurable generation parameters

### Error Handling
- ✅ Syntax error detection
- ✅ Graceful fallbacks
- ✅ Detailed error messages
- ✅ Model loading validation

## 🧪 Testing

Run the integration tests to validate the system:

```bash
cd integration
python test_integration.py
```

The test suite includes:
- Sample function processing
- File processing with multiple functions
- Error handling validation
- Model loading verification

## ⚙️ Configuration

### Model Paths
The system automatically looks for models in:
- `bilstm/bilstm_model.pt` - BiLSTM language model
- `bilstm/bpe_tokenizer.pkl` - BPE tokenizer
- `word2vec/word2vec_model.pt` - Word2Vec model (optional)

### Generation Parameters
- `max_length`: Maximum docstring length (default: 100)
- `device`: CUDA or CPU (auto-detected)
- `context_length`: Input context length (default: 20 tokens)

## 🔍 Model Information

The system provides detailed model information:
- Device usage (CUDA/CPU)
- Model loading status
- Vocabulary size
- Model architecture details

## 📝 Example Output

**Input:**
```python
def calculate_discount(price, discount_percent):
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    return final_price
```

**Generated Documentation:**
```markdown
# Function: calculate_discount

## Summary
Function 'calculate_discount' takes parameters: price, discount_percent

## Parameters
- `price`: Parameter description
- `discount_percent`: Parameter description

## Generated Docstring
"""
Calculate discount amount and final price
"""
```

## 🚨 Troubleshooting

### Common Issues

1. **Models not found**: Ensure all model files are in correct directories
2. **CUDA errors**: System will fallback to CPU automatically
3. **Import errors**: Check Python path and dependencies
4. **Memory issues**: Reduce batch size or use CPU

### Debug Mode

Enable detailed error messages:
```python
generator = DocumentationGenerator()
result = generator.generate_from_string(code)
if result['status'] == 'error':
    print(result['traceback'])
```

## 🔮 Future Enhancements

- Support for more programming languages
- Advanced docstring templates
- Integration with IDEs
- Batch processing optimization
- Custom model fine-tuning

## 📄 License

This project is part of the GenerativeAI_Project educational system.
