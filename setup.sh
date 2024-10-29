conda install -c pytorch faiss-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install pymupdf 
pip install transformers torch einops
pip install 'numpy<2'
pip install flash-attn --no-build-isolation
pip install pytest tiktoken