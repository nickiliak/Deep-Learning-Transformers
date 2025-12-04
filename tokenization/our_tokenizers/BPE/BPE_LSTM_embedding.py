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
            max_length: Max sequence length
            lstm_checkpoint_path: Path to trained LSTM checkpoint (default: models/LSTM/lstm_bpe_best.pt)
        """
        print(f"--- Loading BPE-LSTM Embedder (TRAINED LSTM) ---")

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("Device:", self.device)

        # -----------------------------
        # Load your custom BPE tokenizer
        # -----------------------------
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(bpe_model_path)

        # Build vocab & pad token
        vocab = self.tokenizer.build_vocab()
        vocab_size = max(vocab.keys()) + 1
        self.pad_id = vocab_size
        self.vocab_size = vocab_size + 1

        self.max_length = max_length

        # -----------------------------
        # Load trained LSTM encoder
        # -----------------------------
        if lstm_checkpoint_path is None:
            # Default to best checkpoint
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
    # Encode + pad
    # -----------------------------------------------------------
    def _encode_and_pad(self, texts: List[str]):
        all_ids = []
        for t in texts:
            ids = self.tokenizer.encode(t)

            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            else:
                ids = ids + [self.pad_id] * (self.max_length - len(ids))

            all_ids.append(ids)

        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
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
    # Batch embeddings
    # -----------------------------------------------------------
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        input_ids, mask = self._encode_and_pad(texts)

        with torch.no_grad():
            # LSTM forward pass
            logits, hidden = self.encoder(input_ids)
            # Use LSTM output (before final projection layer)
            # logits shape: (batch, seq, hidden_dim) from LSTM, then projected to vocab_size
            # We want the LSTM hidden states before projection
            # Re-run without projection by accessing LSTM directly
            embedded = self.encoder.dropout(self.encoder.embedding(input_ids))
            lstm_output, _ = self.encoder.lstm(embedded)
            lstm_output = self.encoder.dropout(lstm_output)
            # Now pool the LSTM output
            pooled = self._mean_pool(lstm_output, mask)

        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
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
