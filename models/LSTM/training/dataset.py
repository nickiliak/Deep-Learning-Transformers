"""
Dataset for language modeling
"""

import torch
from torch.utils.data import Dataset
import json


class LanguageModelingDataset(Dataset):
    """
    Dataset that creates input/target pairs for next token prediction
    
    Input:  [token_1, token_2, token_3, ..., token_n]
    Target: [token_2, token_3, token_4, ..., token_n+1]
    """
    
    def __init__(self, tokenizer, texts, seq_length=128, stride=64):
        """
        Args:
            tokenizer: Tokenizer with encode() method
            texts: List of text strings
            seq_length: Length of each training sequence
            stride: How much to slide window (smaller = more overlap)
        """
        self.seq_length = seq_length
        self.data = []
        
        print(f"Creating dataset with seq_length={seq_length}, stride={stride}...")
        
        for text_idx, text in enumerate(texts):
            if text_idx % 100 == 0:
                print(f"  Processing text {text_idx}/{len(texts)}")
            
            # Tokenize
            tokens = tokenizer.encode(text)
            
            # Create sliding windows
            for i in range(0, len(tokens) - seq_length - 1, stride):
                input_seq = tokens[i:i + seq_length]
                target_seq = tokens[i + 1:i + seq_length + 1]
                
                # Only add if we have full sequences
                if len(input_seq) == seq_length and len(target_seq) == seq_length:
                    self.data.append((input_seq, target_seq))
        
        print(f"✅ Created {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def load_documents_from_jsonl(file_path, max_docs=None):
    """Load documents from JSONL file"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            if line.strip():
                doc = json.loads(line)
                # Combine title and text
                text = f"{doc.get('title', '')} {doc.get('text', '')}"
                texts.append(text)
    return texts
