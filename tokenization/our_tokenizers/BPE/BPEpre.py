import torch
from typing import List
from transformers import AutoTokenizer, AutoModel


class BPEPretrainedEmbedder:
    """   
      - Tokenizer: model's own BPE tokenizer (e.g. roberta-base)
      - Encoder: model's own transformer encoder
      - Pooling: mean pooling over last_hidden_state
      - Output: L2-normalized embedding (dim = hidden_size, e.g. 768)

      - generate_embedding(text) -> List[float]
      - generate_embeddings_batch(texts) -> List[List[float]]
      - embedding_dimension property
    """

    def __init__(self, model_id: str = "roberta-base", max_length: int = 512):
        print(f"--- Loading pretrained BPE model: {model_id} ---")

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self._hidden_size = self.model.config.hidden_size  # e.g. 768

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling over the sequence dimension, masking out padding.
        """
        token_embeddings = model_output.last_hidden_state            # (B, T, D)
        input_mask_expanded = attention_mask.unsqueeze(-1).float()   # (B, T, 1)

        # Zero-out pad token embeddings
        masked_embeddings = token_embeddings * input_mask_expanded

        # Sum and divide by number of valid tokens
        sum_embeddings = masked_embeddings.sum(dim=1)                # (B, D)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)    # (B, 1)

        return sum_embeddings / sum_mask

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of texts into L2-normalized sentence embeddings.
        """
        # Tokenize with the model's own BPE tokenizer
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        sentence_embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().tolist()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Encode a single text into a sentence embedding.
        """
        return self.generate_embeddings_batch([text])[0]

    @property
    def embedding_dimension(self) -> int:
        return self._hidden_size