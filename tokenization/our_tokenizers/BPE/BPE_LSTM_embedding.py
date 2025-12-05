from .BPE_tokenization import CustomBPETokenizer
import torch
import torch.nn as nn 
from typing import List
from transformers import RobertaModel
import os
VOCAB_PATH = "tokenization\\vocabularies\\CustomBPETokenizer.json"

class BPELSTMEmbedder:
    """
        - generate_embedding(text) -> List[float]
        - generate_embeddings_batch(texts) -> List[List[float]]
        - embedding_dimension property

        BPE Embedder using a custom BPE tokenizer and a trained LSTM encoder.
    """

    def __init__(
        self,
        bpe_model_path: str = VOCAB_PATH,
        max_length: int = 512,
        lstm_checkpoint_path: str = None,
    ):
        """
        Args:
            bpe_model_path: Path to BPE tokenizer
            max_length: Max sequence length (hard limit for truncation only)
            lstm_checkpoint_path: Path to trained LSTM checkpoint (default: models/LSTM/lstm_bpe_best.pt)
        """
        print(f"--- Loading BPE-LSTM Embedder (TRAINED LSTM) ---")

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("Device:", self.device)

        # Load your custom BPE tokenizer
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(bpe_model_path)

        # Build vocab & pad token
        vocab = self.tokenizer.build_vocab()
        vocab_size = max(vocab.keys()) + 1
        self.pad_id = vocab_size
        self.vocab_size = vocab_size + 1

        self.max_length = max_length

        # Load trained LSTM encoder
        if lstm_checkpoint_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            lstm_checkpoint_path = os.path.join(repo_root, "models", "LSTM", "lstm_bpe_best.pt")
        
        print(f"Loading trained LSTM from: {lstm_checkpoint_path}")
        
        # Import LSTM model
        import sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from models.LSTM.training.lstm_model import SimpleLSTM_LM
        
        # Load checkpoint
        checkpoint = torch.load(lstm_checkpoint_path, map_location=self.device)
        
        # Create model
        self.encoder = SimpleLSTM_LM(
            vocab_size=self.vocab_size,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        )
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.hidden = 256  # LSTM hidden dim
        print(f"✅ Loaded trained LSTM (embedding dimension: {self.hidden})")

    # -----------------------------------------------------------
    # Encode + pad with DYNAMIC PADDING (critical for performance!)
    # -----------------------------------------------------------
    def _encode_and_pad(self, texts: List[str]):
        """
        Encode texts with BPE and pad to BATCH MAX LENGTH, not global max_length.
        This is critical for performance - if batch has short texts (avg 50 tokens),
        we pad to 50, not 512. This gives 10x speedup on small batches!
        """
        # 1. Encode all and truncate to hard limit
        encoded = []
        for t in texts:
            ids = self.tokenizer.encode(t)
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            encoded.append(ids)
        
        # 2. Find batch max length (not global!)
        batch_max_len = max(len(ids) for ids in encoded) if encoded else 1
        
        # 3. Pad to batch max only
        padded = []
        for ids in encoded:
            pad_len = batch_max_len - len(ids)
            padded.append(ids + [self.pad_id] * pad_len)
        
        # 4. Create tensors on GPU directly
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.pad_id).long()

        return input_ids, attention_mask

    # -----------------------------------------------------------
    # Mean pooling (same as CANINE)
    # -----------------------------------------------------------
    def _mean_pool(self, last_hidden, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    # -----------------------------------------------------------
    # Batch embeddings - OPTIMIZED FOR SPEED
    # -----------------------------------------------------------
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for batch of texts.
        CRITICAL OPTIMIZATIONS:
        1. Dynamic padding (batch_max_len, not global max_length)
        2. Skip final linear layer (not needed for embeddings)
        3. Keep on GPU until final output
        4. Use torch.no_grad() context for inference efficiency
        """
        input_ids, mask = self._encode_and_pad(texts)

        with torch.no_grad():
            # Process on GPU in inference mode
            embedded = self.encoder.dropout(self.encoder.embedding(input_ids))
            lstm_output, _ = self.encoder.lstm(embedded)
            lstm_output = self.encoder.dropout(lstm_output)
            
            # Pool on GPU
            pooled = self._mean_pool(lstm_output, mask)
            
            # Normalize on GPU
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
        # Only move to CPU at the very end for output
        return normalized.cpu().tolist()

    # -----------------------------------------------------------
    # Single embedding
    # -----------------------------------------------------------
    def generate_embedding(self, text: str) -> List[float]:
        return self.generate_embeddings_batch([text])[0]

    # -----------------------------------------------------------
    # Dimension property
    # -----------------------------------------------------------
    @property
    def embedding_dimension(self) -> int:
        return self.hidden
    
    def cleanup(self):
        """Explicitly move model off GPU and clean up memory."""
        if torch.cuda.is_available():
            self.encoder = self.encoder.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
