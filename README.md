# 🧠 Python Code Language Model (BPE + Word2Vec + BiLSTM)

This project explores how **natural language processing (NLP) techniques** can be applied to **source code**.  
We build a pipeline that:  
- **Tokenizes Python code and docstrings** using **Byte Pair Encoding (BPE)**  
- Learns **semantic embeddings** via **Word2Vec**  
- Trains a **BiLSTM Language Model** to predict and generate Python-like code  

Inspired by modern code assistants (like GitHub Copilot), this project demonstrates how smaller, classical models can still capture meaningful programming patterns.

---

## 📂 Project Structure

```
GenerativeAI_Project/
├── bpe/                # BPE tokenizer + trained models
│   ├── bpe_tokenizer.py
│   ├── bpe_code.pkl
│   └── bpe_doc.pkl
├── word2vec/           # Word2Vec embeddings
│   ├── word2vec.py
│   ├── utils.py
│   ├── train_word2vec.py
│   └── word2vec_model.pt
├── bilstm/             # BiLSTM language model
│   ├── bilstm_model.py
│   └── bilstm_model.pt
├── data/               # Dataset (❌ not included in repo, see note below)
│   └── clean_dataset.csv
├── notebooks/          # Analysis & evaluation notebooks
│   ├── bpe_evaluation.ipynb
│   ├── word2vec_evaluation.ipynb
│   ├── bilstm_language_model.ipynb
│   └── bilstm_evaluation.ipynb
├── main.py             # Integrated pipeline (BPE → Word2Vec → BiLSTM)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚠️ Notes on Excluded Files

- The `venv/` folder (virtual environment) is **not included** in this repo. Please create your own environment following the instructions below.  
- The `data/` folder is also **not included** because of dataset size.  
  - You will need to provide your own dataset of Python functions & docstrings.  
  - Expected format: `clean_dataset.csv` with two columns:  
    - `code` → Python function code  
    - `docstring` → corresponding natural language description  

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/ibraheem8887/GenerativeAI_Project.git
cd GenerativeAI_Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Place your dataset in the `data/` folder:
```
data/clean_dataset.csv
```

---

## ▶️ How to Run

### 1. BPE Training & Evaluation
```bash
jupyter notebook notebooks/bpe_evaluation.ipynb
```

### 2. Train Word2Vec
```bash
python word2vec/train_word2vec.py
```

### 3. Evaluate Word2Vec
```bash
jupyter notebook notebooks/word2vec_evaluation.ipynb
```

### 4. Train BiLSTM LM
```bash
jupyter notebook notebooks/bilstm_language_model.ipynb
```

### 5. Evaluate BiLSTM LM
```bash
jupyter notebook notebooks/bilstm_evaluation.ipynb
```

### 6. Run Integrated Pipeline
```bash
python main.py
```

---

## 📊 Results (Summary)

### 🔹 BPE Tokenizer
- Reduced vocab size  
- Low OOV rate  
- Good Jaccard similarity  

### 🔹 Word2Vec
- Learned semantic clusters (e.g., `def ↔ return`, `(` ↔ `)`)  

### 🔹 BiLSTM
- Achieved reasonable perplexity  
- Generated syntactically valid Python snippets  

📌 Detailed metrics, plots, and generated samples are available inside the `notebooks/` folder.
