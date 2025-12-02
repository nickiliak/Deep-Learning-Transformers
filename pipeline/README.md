# Pipeline

This folder contains the main embedding generation pipeline and related tools.

## Files

### `pipeline.ipynb` 
**Main embedding generation notebook.**

- Loads corpus from `data_filtered/corpus_filtered.jsonl`
- Generates embeddings using selected model (ByT5, Canine, or BERT)
- Stores embeddings in PostgreSQL with pgvector
- Configure model by changing `CURRENT_EMBEDDER`, `CURRENT_MODEL_ID`, and `VECTOR_DIMENSION`

**Usage:** Run cells sequentially, change embedder config between runs for different models.

### `groq_inference.py`
**LLM inference using Groq API.**

- Example integration with Groq (qwen3-32b model)
- Reads API key from `.env`
- Not yet connected to retrieval pipeline (future RAG integration)

## Subfolders

### `data_preprocessing/`
- `nq_preprocess.ipynb` - Notebook for filtering/preprocessing Natural Questions dataset

### `examples/`
- `bert.py` - Standalone BERT embedding example
- `bert_embeddings.ipynb` - Interactive BERT embedding notebook

## Workflow

1. Preprocess data (if needed): `data_preprocessing/nq_preprocess.ipynb`
2. Generate embeddings: `pipeline.ipynb` (run 3x for each model)
3. Evaluate: Use scripts in `../tokenization/evaluation/`
