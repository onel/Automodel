# Llama-Embed-Nemotron-8B Training Pipeline

## Overview

[llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) is a versatile text embedding model trained by NVIDIA and optimized for retrieval, reranking, semantic similarity, and classification use cases. This model has robust capabilities for multilingual and cross-lingual text retrieval and is designed to serve as a foundational component in text-based Retrieval-Augmented Generation (RAG) systems. This model achieves state-of-the-art performance on the multilingual [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard as of October 21, 2025.

This guide provides step-by-step instructions to reproduce the training pipeline for the `llama-embed-nemotron-8b` model using [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) framework.

## Reproduction Steps

### 1. Download and Prepare the Dataset

Download and prepare the `nvidia/embed-nemotron-dataset-v1` dataset from [Hugging Face](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1). This dataset is a selected subset of the fine-tuning data used for training the `llama-embed-nemotron-8b` model:

```python
python examples/biencoder/llama_embed_nemotron_8b/data_preparation.py \
    --download-path ./embed_nemotron_dataset_v1
```

This script will download the dataset and prepare it for training. 

### 2. Run Model Finetuning

Run the model finetuning with the specified configuration using 8 GPUs:

```bash
torchrun --nproc-per-node=8 examples/biencoder/finetune.py \
    --config examples/biencoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml
```

The final model checkpoint in Hugging Face format will be stored in `output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated`

## Deployment

After fine-tuning your embedding model, you can deploy it for retrieval and RAG applications or share it with the community.

### Load the Model for Inference

Load the fine-tuned embedding model and use it to encode queries and documents:

```python
import torch
from transformers import AutoTokenizer
from nemo_automodel._transformers import NeMoAutoModelBiencoder

# Load the fine-tuned model
model = NeMoAutoModelBiencoder.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated",
    pooling="avg",
    l2_normalize=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated"
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Encode a query
query = "What is machine learning?"
query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

with torch.inference_mode():
    query_embedding = model.encode(query_inputs, encoder="query")

# Encode documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
]
doc_inputs = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}

with torch.inference_mode():
    doc_embeddings = model.encode(doc_inputs, encoder="passage")

# Compute similarity scores
similarities = torch.matmul(query_embedding, doc_embeddings.T)
print(f"Similarity scores: {similarities}")
```

### Use in Retrieval-Augmented Generation (RAG)

Embedding models are essential for RAG systems. Here's how to use your fine-tuned model for document retrieval:

```python
import torch
import numpy as np
from transformers import AutoTokenizer
from nemo_automodel._transformers import NeMoAutoModelBiencoder

# Load model
model = NeMoAutoModelBiencoder.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated",
    pooling="avg",
    l2_normalize=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Example document corpus
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "Machine learning algorithms learn from data.",
    "The Earth orbits around the Sun.",
]

# Encode documents
doc_inputs = tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}

with torch.inference_mode():
    doc_embeddings = model.encode(doc_inputs, encoder="passage")

# Query
query = "Tell me about programming languages"
query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

with torch.inference_mode():
    query_embedding = model.encode(query_inputs, encoder="query")

# Retrieve top-k documents
similarities = torch.matmul(query_embedding, doc_embeddings.T).cpu().numpy()[0]
top_k = 2
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"Query: {query}")
print(f"\nTop {top_k} relevant documents:")
for idx in top_indices:
    print(f"  - {documents[idx]} (score: {similarities[idx]:.4f})")
```

Expected output:
```
Query: Tell me about programming languages

Top 2 relevant documents:
  - Python is a popular programming language. (score: 0.8523)
  - Machine learning algorithms learn from data. (score: 0.7234)
```

### Batch Encoding for Large-Scale Retrieval

For large document collections, encode documents in batches:

```python
import torch
from transformers import AutoTokenizer
from nemo_automodel._transformers import NeMoAutoModelBiencoder

# Load model
model = NeMoAutoModelBiencoder.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated",
    pooling="avg",
    l2_normalize=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def encode_batch(texts, encoder_type="passage", batch_size=32):
    """Encode texts in batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            embeddings = model.encode(inputs, encoder=encoder_type)
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

# Example: Encode 1000 documents
documents = [f"Document {i} content here" for i in range(1000)]
doc_embeddings = encode_batch(documents, encoder_type="passage", batch_size=32)

print(f"Encoded {len(documents)} documents into embeddings of shape {doc_embeddings.shape}")
```

### Publish to Hugging Face Hub

Share your fine-tuned embedding model with the community by publishing it to the Hugging Face Hub.

#### Prerequisites

1. Install the Hugging Face Hub library:

```bash
pip install huggingface_hub
```

2. Log in to Hugging Face:

```bash
huggingface-cli login
```

#### Upload the Model

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload the checkpoint
api.upload_folder(
    folder_path="output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated",
    repo_id="your-username/llama-embed-nemotron-8b-custom",
    repo_type="model"
)
```

#### Load from the Hub

Once uploaded, others can load your embedding model directly:

```python
from transformers import AutoTokenizer
from nemo_automodel._transformers import NeMoAutoModelBiencoder

# Load from Hub
model = NeMoAutoModelBiencoder.from_pretrained(
    "your-username/llama-embed-nemotron-8b-custom",
    pooling="avg",
    l2_normalize=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/llama-embed-nemotron-8b-custom"
)

# Use for encoding
# ... (inference code)
```

#### Model Card

When publishing your embedding model, include a model card with:
- Training dataset information
- Fine-tuning configuration details
- Evaluation metrics on MTEB benchmarks
- Intended use cases (retrieval, classification, semantic similarity)
- Example usage code for retrieval applications
- Performance characteristics (embedding dimension, inference speed)

## Citation

If you use this model or training pipeline in your research, please cite:

```bibtex
@misc{babakhin2025llamaembednemotron8buniversaltextembedding,
      title={Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks}, 
      author={Yauhen Babakhin and Radek Osmulski and Ronay Ak and Gabriel Moreira and Mengyao Xu and Benedikt Schifferer and Bo Liu and Even Oldridge},
      year={2025},
      eprint={2511.07025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07025}, 
}
```