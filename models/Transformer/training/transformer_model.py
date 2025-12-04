"""
Simple Transformer Language Model
Small-scale model for tokenization comparison (comparable to LSTM)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformer_LM(nn.Module):
    """
    Small-scale Transformer Language Model for next token prediction
    
    Architecture:
    - Embedding layer
    - Positional encoding
    - 3-layer Transformer encoder with causal masking
    - Linear output layer
    - ~3.4M parameters (comparable to LSTM's 2.1M)
    
    Args:
        vocab_size: Size of token vocabulary
        d_model: Dimension of model embeddings (default: 256)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of Transformer layers (default: 3)
        dim_feedforward: Dimension of feedforward network (default: 512)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, 
                 dim_feedforward=512, dropout=0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False  # Post-norm (standard)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        # Embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def _generate_square_subsequent_mask(self, sz, device):
        """
        Generate causal mask to prevent attending to future tokens
        Returns upper triangular matrix of -inf (above diagonal)
        
        Args:
            sz: Sequence length
            device: Device to create mask on
        
        Returns:
            mask: (sz, sz) causal mask
        """
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, input_ids, src_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            src_mask: Optional source mask (seq_length, seq_length)
        
        Returns:
            logits: Predicted logits for next token (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask if not provided
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len, device)
        
        # Embed tokens and scale by sqrt(d_model) (standard Transformer scaling)
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        
        # Pass through Transformer encoder with causal mask
        transformer_out = self.transformer_encoder(embedded, mask=src_mask)
        
        # Project to vocabulary
        logits = self.fc_out(transformer_out)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, start_tokens, max_length=50, temperature=1.0, top_k=None):
        """
        Generate text given starting tokens (autoregressive)
        
        Args:
            start_tokens: Initial token IDs (list or tensor)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            generated_tokens: List of generated token IDs
        """
        self.eval()
        device = next(self.parameters()).device
        
        if isinstance(start_tokens, list):
            tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)
        else:
            tokens = start_tokens.unsqueeze(0) if start_tokens.dim() == 1 else start_tokens
        
        generated = tokens[0].tolist()
        
        with torch.no_grad():
            for _ in range(max_length - len(generated)):
                # Forward pass on current sequence
                logits = self.forward(tokens)
                
                # Get logits for last token
                logits = logits[0, -1, :] / temperature  # (vocab_size,)
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                
                # Append to sequence
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return generated


def create_model_for_tokenizer(tokenizer_type, vocab_size, **kwargs):
    """
    Factory function to create model with appropriate settings
    
    Args:
        tokenizer_type: 'bpe', 'char', or 'byte'
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
    
    Returns:
        SimpleTransformer_LM model
    """
    # Default parameters for small-scale model (comparable to LSTM)
    default_params = {
        'd_model': 256,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.2
    }
    default_params.update(kwargs)
    
    model = SimpleTransformer_LM(vocab_size, **default_params)
    
    print(f"\nCreated Transformer-{tokenizer_type.upper()} model:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Model dimension: {default_params['d_model']}")
    print(f"  Attention heads: {default_params['nhead']}")
    print(f"  Layers: {default_params['num_layers']}")
    print(f"  Feedforward dim: {default_params['dim_feedforward']}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Transformer Language Model...")
    
    # Test with BPE vocab size (2000)
    model_bpe = create_model_for_tokenizer('bpe', vocab_size=2000)
    
    # Test forward pass
    batch_size, seq_length = 4, 32
    dummy_input = torch.randint(0, 2000, (batch_size, seq_length))
    
    logits = model_bpe(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test generation
    start_tokens = [100, 200, 300]
    generated = model_bpe.generate(start_tokens, max_length=20)
    print(f"\nTest generation:")
    print(f"  Start tokens: {start_tokens}")
    print(f"  Generated length: {len(generated)}")
    
    print("\n✅ Model test successful!")
