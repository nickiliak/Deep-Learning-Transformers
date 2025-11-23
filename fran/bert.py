# %% [markdown]
# # BERT EMBEDDINGS

# %%
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# %%

# 1. Configuration
# ----------------
# We use the MiniLM model (BERT distilled).
# Output dimension: 384
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'

print(f"Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

# %%
# 2. Mathematical Logic (Mean Pooling)
# ----------------
def mean_pooling(model_output, attention_mask):
    """
    Collapses the matrix of token vectors into a single sentence vector
    by calculating the weighted average, ignoring padding tokens.
    """
    # The first element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state 
    
    # Expand attention_mask to match the size of embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum the embeddings of valid tokens (mask value = 1)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Count valid tokens (clamping min to 1e-9 to avoid division by zero)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Calculate Mean
    return sum_embeddings / sum_mask

# %%
# 3. Main Inference Function
# ----------------
def get_embedding(text: str):
    # A. Tokenize
    # Convert text to BERT input format (Add [CLS], [SEP], padding)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # B. Model Inference
    with torch.no_grad(): # Disable gradient calculation to save memory
        model_output = model(**encoded_input)

    # C. Pooling
    # Convert token vectors -> Single vector
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # D. Normalization
    # Normalize result (Length = 1) for Cosine Similarity usage
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    # Return as a standard Python list (or numpy array)
    return sentence_embeddings[0].tolist()


# %%
if __name__ == "__main__":
    text = "cat"

    vector = get_embedding(text)

    print("-" * 30)
    print(f"Input Text: '{text}'")
    print(f"Vector Dimension: {len(vector)}") # Should be 384
    print(f"First 5 values: {vector[:5]}")
    print("-" * 30)

# %%



