# A very GPU-friendly RAG implementation

A simple RAG tool for asking things related to a pdf file (in this project, it's about Don Quixote)

This project demonstrates how things can be implemented with GPU-friendly database and models. It uses
- [FAISS](https://github.com/facebookresearch/faiss) for vector database (GPU version)
- [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) for text embedding model
- [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) for rerank model
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) for language model
  
I use a NVIDIA A30 GPU to run this. 

## Setup

Python 3.10, miniconda

Install required packages, please see `setup.sh` for extra information
```bash
bash setup.sh
```

## Run

It is strongly advised to reach each .py file before running any command. 
By doing so, you get to understand the project more.

At root project, to setup the vector database
```bash
python setup_db.py
```

At root project, to query something then have a language model answer
```bash
python rag.py
```

## Great researcher/developer-friendly RAG frameworks

- [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- [neuml/txtai](https://github.com/neuml/txtai)
- [SylphAI-Inc/AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)