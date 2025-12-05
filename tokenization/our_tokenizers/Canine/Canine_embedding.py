import torch
from typing import List
from transformers import AutoModel
from .Canine_tokenization import CanineTokenizer


class CanineEmbedder:
    """
    Embedder using CANINE (Character Architecture with No Tokenization In Neural Encoders).
    
    CANINE operates directly on Unicode characters without explicit tokenization.
    - Model: google/canine-s (or canine-c for more capacity)
    - Output dimension: 768 (hidden size)
    - Character-level: Excellent for rare words, code-switching, multilingual text
    """
    
    def __init__(self, model_id: str = "google/canine-s"):
        print(f"--- Loading CANINE Model: {model_id} ---")
        
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        canine_tokenizer = CanineTokenizer(model_id)
        self.tokenizer = canine_tokenizer.tokenizer
        self.model = AutoModel.from_pretrained(
            model_id,
            use_safetensors=True  # ADD THIS LINE
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling: Average all token embeddings (ignoring padding).
        """
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
            List of 768 floats (normalized embedding)
        """
        # A. Tokenize (CANINE uses character-level encoding)
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=2048,  # CANINE supports longer sequences
            return_tensors='pt'
        ).to(self.device)
        
        # B. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
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
            List of embedding vectors (each is 768 floats)
        """
        # A. Tokenize all texts at once (automatic padding to longest in batch)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        ).to(self.device)
        
        # B. Batch Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # C. Pooling for all sequences
            sentence_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # D. Normalize all embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Return as list of lists
        return sentence_embeddings.cpu().tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Return the output embedding dimension."""
        return self.model.config.hidden_size  # 768 for canine-s
    
    def cleanup(self):
        """Explicitly move model off GPU and clean up memory."""
        if torch.cuda.is_available():
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
