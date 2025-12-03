"""
Simple LSTM Language Model
Small-scale model for tokenization comparison
"""

import torch
import torch.nn as nn


class SimpleLSTM_LM(nn.Module):
    """
    Small-scale LSTM Language Model for next token prediction
    
    Architecture:
    - Embedding layer
    - 2-layer LSTM (256 hidden units each)
    - Linear output layer
    - ~2M parameters (truly "small-scale")
    
    Args:
        vocab_size: Size of token vocabulary
        embedding_dim: Dimension of token embeddings (default: 256)
        hidden_dim: Hidden dimension of LSTM (default: 256)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        # Embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, input_ids, hidden=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            hidden: Previous hidden state (optional)
        
        Returns:
            logits: Predicted logits for next token (batch_size, seq_length, vocab_size)
            hidden: New hidden state
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden = self.lstm(embedded)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        logits = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        
        return logits, hidden
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, start_tokens, max_length=50, temperature=1.0, top_k=None):
        """
        Generate text given starting tokens
        
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
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length - len(generated)):
                # Get predictions for last token
                logits, hidden = self.forward(tokens, hidden)
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
                tokens = next_token.unsqueeze(0).unsqueeze(0)
        
        return generated


def create_model_for_tokenizer(tokenizer_type, vocab_size, **kwargs):
    """
    Factory function to create model with appropriate settings
    
    Args:
        tokenizer_type: 'bpe', 'char', or 'byte'
        vocab_size: Size of vocabulary
        **kwargs: Additional model parameters
    
    Returns:
        SimpleLSTM_LM model
    """
    # Default parameters for small-scale model
    default_params = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2
    }
    default_params.update(kwargs)
    
    model = SimpleLSTM_LM(vocab_size, **default_params)
    
    print(f"\nCreated LSTM-{tokenizer_type.upper()} model:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dim: {default_params['embedding_dim']}")
    print(f"  Hidden dim: {default_params['hidden_dim']}")
    print(f"  Layers: {default_params['num_layers']}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing LSTM Language Model...")
    
    # Test with BPE vocab size (2000)
    model_bpe = create_model_for_tokenizer('bpe', vocab_size=2000)
    
    # Test forward pass
    batch_size, seq_length = 4, 32
    dummy_input = torch.randint(0, 2000, (batch_size, seq_length))
    
    logits, hidden = model_bpe(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Hidden states: {len(hidden)}")
    
    # Test generation
    start_tokens = [100, 200, 300]
    generated = model_bpe.generate(start_tokens, max_length=20)
    print(f"\nTest generation:")
    print(f"  Start tokens: {start_tokens}")
    print(f"  Generated length: {len(generated)}")
    
    print("\n✅ Model test successful!")
