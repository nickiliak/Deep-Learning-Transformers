import torch
from typing import List
from transformers import AutoModel
from .ByT5_tokenization import ByT5Tokenizer

class ByT5Embedder:
    """
    Embedder using ByT5 (Byte-level T5) model.
    
    ByT5 operates directly on UTF-8 bytes without traditional tokenization.
    - Model: google/byt5-small
    - Output dimension: 1472 (encoder hidden size)
    - Character-aware: Better for rare words, typos, multilingual text
    """
    
    def __init__(self, model_id: str = "google/byt5-small"):
        print(f"--- Loading ByT5 Model: {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Use the underlying HuggingFace tokenizer from your wrapper
        byt5_tokenizer = ByT5Tokenizer(model_id)
        self.tokenizer = byt5_tokenizer.tokenizer  # Get the AutoTokenizer inside
        self.model = AutoModel.from_pretrained(
            model_id,
            use_safetensors=True  # ADD THIS LINE
        ).to(self.device)
        
        # ByT5 is encoder-decoder; we only use encoder for embeddings
        self.model.eval()  # Set to evaluation mode
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling: Average all token embeddings (ignoring padding).
        """
        # For T5, use encoder's last hidden state
        token_embeddings = model_output.last_hidden_state
        
        # Expand mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum valid token embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for input text.
        
        Args:
            text: Input string
            
        Returns:
            List of 1472 floats (normalized embedding)
        """
        # A. Tokenize (ByT5 works on bytes, not subwords)
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,  # Limit sequence length
            return_tensors='pt'
        ).to(self.device)
        
        # B. Inference (encoder only)
        with torch.no_grad():
            # For T5, we need to use the model's encoder directly with proper method
            outputs = self.model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
        
        # C. Pooling (collapse sequence to single vector)
        sentence_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # D. Normalize (L2 normalization for cosine similarity)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Return as Python list
        return sentence_embeddings[0].cpu().tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts simultaneously (GPU batching).
        
        Args:
            texts: List of input strings
            
        Returns:
            List of embedding vectors (each is 1472 floats)
        """
        # A. Tokenize all texts at once (automatic padding to longest in batch)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # B. Batch Inference
        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
        
        # C. Pooling for all sequences
        sentence_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # D. Normalize all embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Return as list of lists
        return sentence_embeddings.cpu().tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Return the output embedding dimension."""
        return self.model.config.d_model  # 1472 for byt5-small
