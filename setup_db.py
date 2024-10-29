from pathlib import Path

import faiss
import numpy as np
import pymupdf
import torch
from tqdm import tqdm
from transformers import AutoModel

embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
if torch.cuda.is_available():
    embedding_model = embedding_model.to("cuda")
    print("Move model to GPU")

embedding_size = 1024
embedding_db = np.empty((0, embedding_size), dtype=np.float32)

file_path = Path("data/very-short-english-stories-for-children-and-esl-students-www.learnenglishteam.com_.pdf")
with pymupdf.open(file_path) as pdf:
    for page in tqdm(pdf):
        content = page.get_text()
        embedding = embedding_model.encode([content])
        embedding_db = np.append(embedding_db, embedding, axis=0)

embedding_index = faiss.IndexFlatL2(embedding_size)
embedding_index.add(embedding_db)
faiss.write_index(embedding_index, "data/embedding.index")
