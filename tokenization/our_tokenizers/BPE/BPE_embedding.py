from .BPE_tokenization import CustomBPETokenizer
import torch
import torch.nn as nn 
from typing import List
from transformers import RobertaModel
VOCAB_PATH = "tokenization\\vocabularies\\CustomBPETokenizer.json"

class BPEEmbedder:
    """
        - generate_embedding(text) -> List[float]
        - generate_embeddings_batch(texts) -> List[List[float]]
        - embedding_dimension property

        BPE Embedder using a custom BPE tokenizer and a pretrained Roberta encoder.
    """

    def __init__(
        self,
        bpe_model_path: str = VOCAB_PATH,
        max_length: int = 512,
        model_name: str = "roberta-base",
    ):
        print(f"--- Loading BPE Embedder (pretrained {model_name}) ---")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # Load pretrained encoder
        # -----------------------------
        print(f"Loading pretrained encoder: {model_name}")
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Replace tokenizer's embedding with new embedding tied to your vocab
        hidden = self.encoder.config.hidden_size
        self.encoder.embeddings.word_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=hidden,
        ).to(self.device)

        print(f"Embedding dimension: {hidden}")
        self.hidden = hidden

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
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=mask,
            )

        pooled = self._mean_pool(outputs.last_hidden_state, mask)
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