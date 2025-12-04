# Models

This directory contains custom-trained language models for investigating tokenization strategies in document retrieval tasks.

## Overview

We train small-scale language models (2-3M parameters) from scratch to compare architectural choices and tokenization strategies against pretrained baselines.

## Models

### 1. LSTM Language Model (`LSTM/`)
- **Architecture:** 2-layer LSTM with 256 hidden units per layer
- **Parameters:** 2.1M trainable parameters
- **Tokenization:** Custom BPE tokenizer (2K vocabulary)
- **Training Data:** 5,000 Wikipedia documents from Natural Questions
- **Training Objective:** Next-token prediction (language modeling)
- **Performance:** Perplexity ~48, BPC 3.2
- **Retrieval:** 15% Recall@10, 51ms average latency
- **Key Files:**
  - `training/lstm_model.py` - Model architecture
  - `training/dataset.py` - Data loading and preprocessing
  - `training/train_bpe_lstm.py` - Training script
  - `lstm_bpe_best.pt` - Best checkpoint (lowest validation loss)
  - `lstm_bpe_final.pt` - Final checkpoint after 10 epochs

### 2. Transformer Language Model (`Transformer/`) *[To be implemented]*
- **Architecture:** 3-layer Transformer encoder with 4 attention heads
- **Parameters:** ~3.4M trainable parameters
- **Tokenization:** Custom BPE tokenizer (same as LSTM)
- **Training Data:** Same 5,000 Wikipedia documents
- **Training Objective:** Next-token prediction (language modeling)
- **Purpose:** Compare Transformer vs LSTM architectures at small scale

## Training Setup

**Common Configuration:**
- Sequence length: 128 tokens
- Batch size: 32
- Optimizer: Adam with learning rate scheduling
- Loss: CrossEntropyLoss
- Epochs: 10
- Hardware: GPU-enabled (CUDA/MPS)

## Usage

### Training LSTM
```bash
cd LSTM/training
python train_bpe_lstm.py
```

### Using Trained Models for Retrieval
Models are automatically loaded by their respective embedders in `tokenization/our_tokenizers/`:
- `BPE/BPE_embedding.py` - Uses trained LSTM
- Generates 256-dimensional embeddings via mean pooling

## Evaluation

Models are evaluated on document retrieval using:
- **Dataset:** Natural Questions (7,247 corpus documents, 100 test queries)
- **Metrics:** Recall@K, MRR, Precision@K, query latency
- **Baseline Comparisons:** BERT-MiniLM, Canine, RoBERTa

See `tokenization/evaluation/` for evaluation scripts and results.

## Key Findings

1. **Small-scale training works:** 2M parameter LSTM achieves 15% Recall@10, matching 125M parameter RoBERTa
2. **Architecture comparison:** Direct LSTM vs Transformer comparison with identical training setup
3. **Training objective matters:** Task-specific training (BERT: 100%) >> general language modeling (15%)
4. **Efficiency advantage:** Custom LSTM has fastest inference (51ms) among all tested models
