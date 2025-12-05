# Language Model Architectures

This directory contains two small-scale language models trained for next-token prediction with custom tokenizers. Both models were designed to be comparable in scale (~2-3M parameters) for fair evaluation.

---

## LSTM Language Model

### Architecture Overview

**SimpleLSTM_LM** is a recurrent neural network (RNN) based language model that uses LSTM cells to capture sequential dependencies in text.

#### Key Components:

1. **Token Embedding Layer**
   - Converts token IDs to dense vector representations
   - Dimension: 256
   - Initialization: Uniform distribution [-0.1, 0.1]

2. **2-Layer LSTM Stack**
   - Each layer: 256 hidden units
   - Processes sequences bidirectionally through time
   - Includes dropout (0.2) between layers for regularization
   - Maintains hidden state across sequence for long-range dependencies

3. **Output Projection**
   - Linear layer mapping hidden states to vocabulary logits
   - Inputs: 256-dim LSTM output
   - Outputs: vocab_size logits per token

#### Design Philosophy:

- **Recurrent Processing**: Processes tokens sequentially, maintaining state
- **Hidden State Memory**: Can leverage information from entire input sequence via hidden state
- **Weight Sharing**: Same LSTM weights apply to every timestep
- **Stateful Generation**: Can continue generating from previous hidden state

#### Parameters:
- **Total Parameters**: ~2.1M
- **Embedding Parameters**: 512K (vocab_size × 256)
- **LSTM Parameters**: 1.4M
- **Output Layer**: 512K

#### Training Performance:
- **Perplexity (PPL)**: 48
- **Bits Per Character (BPC)**: 3.2
- **Epochs**: 10

#### Inference Characteristics:
- **Latency**: ~51ms on GPU (batch processing)
- **Memory**: ~850MB on GPU
- **Speed Profile**: Constant per-token generation time
- **Best For**: Streaming/online generation

---

## Transformer Language Model

### Architecture Overview

**SimpleTransformer_LM** is an attention-based language model that uses a Transformer encoder with causal masking to prevent attending to future tokens.

#### Key Components:

1. **Token Embedding Layer**
   - Converts token IDs to dense vectors
   - Dimension: 256
   - Initialization: Uniform distribution [-0.1, 0.1]
   - Scaled by √d_model during forward pass (standard Transformer scaling)

2. **Positional Encoding**
   - Sinusoidal positional encodings added to embeddings
   - Encodes absolute position information without learnable parameters
   - Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - Allows attention to be position-aware

3. **3-Layer Transformer Encoder**
   - Each layer contains:
     - **Multi-Head Self-Attention**: 4 attention heads
     - **Feed-Forward Network**: 512 hidden units
     - **Layer Normalization**: Post-norm (after residual connection)
     - **Dropout**: 0.2 for regularization
   - **Causal Masking**: Prevents attending to future tokens (upper triangular -inf mask)

4. **Output Projection**
   - Linear layer from d_model (256) to vocabulary logits
   - Generates predictions for all tokens in parallel

#### Design Philosophy:

- **Parallel Processing**: All tokens processed simultaneously (more efficient than RNN)
- **Attention Mechanism**: Can learn complex long-range dependencies without sequential bottleneck
- **Position Awareness**: Positional encodings preserve sequence order information
- **Causal Masking**: Prevents information leakage from future tokens during training/generation

#### Parameters:
- **Total Parameters**: ~3.4M
- **Embedding Parameters**: 512K (vocab_size × 256)
- **Positional Encoding**: 1.28M (fixed, not trainable)
- **Transformer Layers**: 1.5M
- **Output Layer**: 512K

#### Training Performance:
- **Architecture**: 3-layer encoder with 4 attention heads
- **Epochs**: 50 (retrained after initial underperformance)
- **Note**: Scaled longer than LSTM for convergence

#### Inference Characteristics:
- **Latency**: ~45ms on GPU (batch processing with dynamic padding)
- **Memory**: ~950MB on GPU
- **Speed Profile**: All-at-once processing, benefits from batch processing
- **Best For**: Batch inference, high-throughput generation

---

## Comparative Analysis

| Aspect | LSTM | Transformer |
|--------|------|-------------|
| **Architecture Type** | Recurrent (RNN) | Attention-based (Encoder) |
| **Processing Mode** | Sequential | Parallel |
| **Hidden State** | Explicit (h, c) | Implicit (attention) |
| **Long-Range Dependencies** | Via hidden state (gradient issues) | Via attention (no vanishing gradient) |
| **Position Encoding** | Implicit via order | Explicit sinusoidal |
| **Training Complexity** | Lower | Higher (more compute) |
| **Inference Latency (Single)** | ~51ms | ~45ms |
| **Batch Efficiency** | Constant per-token | High (parallelizable) |
| **Memory Usage** | ~850MB | ~950MB |
| **Parameter Count** | 2.1M | 3.4M |
| **Embedding Dimension** | 256 | 256 |
| **Hidden/Model Dimension** | 256 | 256 |
| **Layers** | 2 layers | 3 layers |
| **Perplexity** | 48 | TBD (50 epochs) |
| **BPC** | 3.2 | TBD |
| **Training Epochs** | 10 | 50 |
| **Best Use Case** | Streaming generation | Batch processing |
| **Gradient Flow** | Prone to vanishing | Stable (no depth issues) |
| **Computation per Token** | Sequential dependency | Independent per position |

---

## Key Differences Explained

### 1. **Processing Paradigm**

**LSTM:**
- Processes tokens one-at-a-time sequentially
- Each token must wait for previous tokens to be processed
- Hidden state accumulates information as it flows through time
- Natural for streaming/online scenarios

**Transformer:**
- Processes all tokens simultaneously (after embedding)
- Attention allows any token to attend to any other token
- Position information explicit rather than implicit
- Ideal for batch processing and offline tasks

### 2. **Long-Range Dependency Modeling**

**LSTM:**
- Relies on hidden state and cell state to propagate information
- Suffers from vanishing/exploding gradients in very long sequences
- Information compressed into fixed-size hidden state (256-dim)

**Transformer:**
- Attention creates direct paths between any two positions
- No information bottleneck
- Can attend to relevant tokens regardless of distance
- Gradient flow is stable across layers

### 3. **Scalability**

**LSTM:**
- Scales poorly with sequence length
- Cannot parallelize across timesteps
- 50x longer training for similar performance (10 vs 50 epochs)

**Transformer:**
- Scales well with batch size (parallelizable)
- Can process longer sequences efficiently
- Requires more training (50 epochs vs 10) for convergence

### 4. **Parameter Distribution**

**LSTM:**
- Parameters concentrated in LSTM cells (1.4M of 2.1M)
- Embedding and output layer relatively small

**Transformer:**
- Parameters spread across attention heads and FFN layers
- More parameter efficiency due to weight sharing in attention

---

## Training Observations

### Why Transformer Needed 50 Epochs vs LSTM's 10?

1. **Increased Model Capacity**: 3.4M vs 2.1M parameters
2. **Attention Complexity**: Learning attention patterns is more complex than LSTM dynamics
3. **Initialization Sensitivity**: Transformer attention is sensitive to initialization
4. **Longer Convergence**: Parallel processing requires more gradient updates for stability

### Performance Metrics

| Model | Perplexity | BPC | Training Epochs | GPU Memory |
|-------|-----------|-----|-----------------|-----------|
| LSTM | 48 | 3.2 | 10 | ~850MB |
| Transformer | TBD | TBD | 50 | ~950MB |

---

## Usage

### LSTM Model

```python
from lstm_model import SimpleLSTM_LM

# Create model
model = SimpleLSTM_LM(vocab_size=2000, embedding_dim=256, hidden_dim=256, num_layers=2)

# Forward pass
input_ids = torch.randint(0, 2000, (batch_size, seq_length))
logits, hidden = model(input_ids)

# Generation
generated_ids = model.generate(start_tokens=[1, 2, 3], max_length=100)
```

### Transformer Model

```python
from transformer_model import SimpleTransformer_LM

# Create model
model = SimpleTransformer_LM(vocab_size=2000, d_model=256, nhead=4, num_layers=3)

# Forward pass
input_ids = torch.randint(0, 2000, (batch_size, seq_length))
logits = model(input_ids)

# Generation
generated_ids = model.generate(start_tokens=[1, 2, 3], max_length=100)
```

---

## Integration with Embedding Pipelines

Both models are integrated with custom tokenizers to generate contextualized embeddings:

- **BPE_LSTM_embedding.py**: LSTM + custom BPE tokenizer → 256-dim embeddings
- **BPE_Transformer_embedding.py**: Transformer + custom BPE tokenizer → 256-dim embeddings

These embeddings are used in the retrieval pipeline for document ranking.

---

## Model Files

- `LSTM/training/lstm_model.py` - LSTM architecture definition
- `LSTM/lstm_bpe_final.pt` - Trained LSTM weights
- `Transformer/training/transformer_model.py` - Transformer architecture definition  
- `Transformer/transformer_bpe_final.pt` - Trained Transformer weights

---

## Performance Optimization

Both models have been optimized for inference:

1. **Dynamic Padding**: Pads sequences to actual max length in batch (not global 512)
2. **GPU Grouping**: All operations stay on GPU within `torch.no_grad()` context
3. **Batch Processing**: Optimized for parallel inference across multiple texts
4. **Memory Efficiency**: Minimal GPU-CPU transfers (only final output moved to CPU)

Expected speedup from optimizations:
- LSTM: 5-10x faster on short text batches
- Transformer: 5-10x faster on short text batches
- Tokenizer: 100x faster (O(n²) → O(n) greedy algorithm)

---

## Future Improvements

1. **Longer Training**: Continue training Transformer beyond 50 epochs
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, layer count
3. **Mixed Precision Training**: Use fp16 for faster training on newer GPUs
4. **Knowledge Distillation**: Transfer knowledge from large models to optimize scale
5. **Quantization**: Reduce model size while maintaining performance
