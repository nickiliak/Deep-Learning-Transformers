import torch
from typing import List
from transformers import AutoTokenizer, AutoModel

class BertEmbedder:
    """
    Embedder using BERT architecture (specifically MiniLM for semantic search).
    
    - Default Model: sentence-transformers/all-MiniLM-L6-v2
    - Output dimension: 384
    - Token limit: 512 tokens (standard BERT limit)
    """
    
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"--- Loading BERT Model: {model_id} ---")
        
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling: Average all token embeddings (ignoring padding).
        Identical logic to the original notebook script.
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
        """
        # A. Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,  # BERT standard limit
            return_tensors='pt'
        ).to(self.device)
        
        # B. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # C. Pooling
        sentence_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # D. Normalize
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Return as Python list
        return sentence_embeddings[0].cpu().tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts simultaneously (GPU batching).
        Crucial for pipeline speed.
        """
        # A. Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # B. Batch Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # C. Pooling
        sentence_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # D. Normalize
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Return as list of lists
        return sentence_embeddings.cpu().tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Return the output embedding dimension (384 for MiniLM)."""
        return self.model.config.hidden_size
    
    def cleanup(self):
        """Explicitly move model off GPU and clean up memory."""
        if torch.cuda.is_available():
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()