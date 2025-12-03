from .BPE_tokenization import CustomBPETokenizer
import torch
import torch.nn as nn 
from typing import List
VOCAB_PATH = "tokenization\\vocabularies\\CustomBPETokenizer.json"

def load_CustomBPETokenizer(model_path: str) -> CustomBPETokenizer:
    """
    Load your trained BPE tokenizer from JSON.
    """
    tokenizer = CustomBPETokenizer()
    tokenizer.load(model_path)
    return tokenizer



class BPEEmbedder(nn.Module):
    """
    Embedder using your custom BPE tokenizer + a lightweight LSTM encoder.

    Produces a fixed-size embedding (default: 768 dimensions),
    just like the CANINE embedder, enabling homogeneous downstream processing.
    """

    def __init__(
        self,
        bpe_model_path: str = VOCAB_PATH,
        d_model: int = 768,
        max_length: int = 2048,
    ):
        super().__init__()

        print("--- Loading BPE tokenizer and building embedder ---")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # -----------------------------
        # Load trained BPE tokenizer
        # -----------------------------
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(bpe_model_path)

        # -----------------------------
        # Build vocabulary & embedding
        # -----------------------------
        vocab = self.tokenizer.build_vocab()
        vocab_size = max(vocab.keys()) + 1
        self.pad_id = vocab_size          # reserve last id for PAD
        num_embeddings = vocab_size + 1   # +1 for PAD

        self.d_model = d_model
        self.max_length = max_length

        # Learnable embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
        )

        # -----------------------------
        # LSTM encoder (1-layer)
        # -----------------------------
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.to(self.device)
        self.eval()
    def _encode_and_pad(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenize and pad/truncate to fixed max_length.
        Returns: LongTensor (batch_size, max_length)
        """
        all_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
            else:
                ids = ids + [self.pad_id] * (self.max_length - len(ids))
            all_ids.append(ids)

        return torch.tensor(all_ids, dtype=torch.long, device=self.device)
    def _mean_pooling(self, embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(embeddings * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / lengths
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a batch of texts into fixed-size embeddings.
        Output: List of 768-dim vectors (float lists)
        """
        self.eval()

        # 1. Encode + pad
        input_ids = self._encode_and_pad(texts)
        attention_mask = (input_ids != self.pad_id).long()

        # 2. Pass through embedding + LSTM
        with torch.no_grad():
            token_embs = self.embedding(input_ids)
            outputs, _ = self.lstm(token_embs)

        # 3. Mean pooling
        pooled = self._mean_pooling(outputs, attention_mask)

        # 4. Normalize
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return normed.cpu().tolist()

    # -----------------------------------------------------------
    # Single text embedding
    # -----------------------------------------------------------
    def generate_embedding(self, text: str) -> List[float]:
        return self.generate_embeddings_batch([text])[0]

    # -----------------------------------------------------------
    # Property: embedding dimensionality
    # -----------------------------------------------------------
    @property
    def embedding_dimension(self) -> int:
        return self.d_model