from pathlib import Path

import faiss
import pymupdf
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, pipeline

embedding_index = faiss.read_index("data/embedding.index")
embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)
if torch.cuda.is_available():
    embedding_model = embedding_model.to("cuda").eval()
    rerank_model = rerank_model.to("cuda").eval()
    print("Move model(s) to GPU")

print("Query:")
query = """Summarize the story of THE GREEDY CLOUD
"""
print(query)
query_embedding = embedding_model.encode([query])

# Simple search
max_search_results = 5
source_distances, batch_source_ids = embedding_index.search(query_embedding, max_search_results)
source_ids = batch_source_ids[0]
context_list = []
file_path = Path("data/very-short-english-stories-for-children-and-esl-students-www.learnenglishteam.com_.pdf")
with pymupdf.open(file_path) as pdf:
    pdf.select(source_ids.tolist())
    for page in pdf:
        context = page.get_text()
        context_list.append(context)

print(f"Closest page {source_ids[0]+1}")

# Simple rerank
rerank_scores = rerank_model.compute_score([[query, i] for i in context_list])
max_rerank_score_index = rerank_scores.index(max(rerank_scores))
print(f"After reranked, closest page {source_ids[max_rerank_score_index]+1}")

# Simple chat
instruction = f"""Do the following query, given the context below.

Query:
{query}

Context:
{context_list[max_rerank_score_index]}
"""

messages = [
    {"role": "system", "content": "Answer in simple English"},
    {"role": "user", "content": instruction},
]

model_id = "meta-llama/Llama-3.2-3B-Instruct"
chat_pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
)
outputs = chat_pipe(messages, max_new_tokens=1024)

print("Answer:")
content = outputs[0]["generated_text"][-1]["content"]
print(content)
