"""
Tokenization Analysis Script
Analyzes tokenization efficiency, compression, vocabulary coverage across different tokenizers
"""

import json
import os
import sys
from typing import List, Dict, Tuple
from collections import Counter
import statistics
import csv
from datetime import datetime

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tokenization.our_tokenizers.ByT5.ByT5_tokenization import ByT5Tokenizer
from tokenization.our_tokenizers.Canine.Canine_tokenization import CanineTokenizer
from tokenization.our_tokenizers.BPE.BPE_tokenization import CustomBPETokenizer
from transformers import AutoTokenizer

# Configuration
CORPUS_PATH = os.path.join(repo_root, "data_filtered", "corpus_filtered.jsonl")
QUERIES_PATH = os.path.join(repo_root, "data_filtered", "queries_filtered.jsonl")
MAX_SAMPLES = 100  # Analyze first N documents for speed


def load_documents(path: str, max_docs: int = None) -> List[str]:
    """Load document texts from JSONL"""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            if line.strip():
                doc = json.loads(line)
                # Combine title and text
                text = f"{doc.get('title', '')} {doc.get('text', '')}"
                texts.append(text)
    return texts


def count_words(text: str) -> int:
    """Simple word count (split by whitespace)"""
    return len(text.split())


def count_characters(text: str) -> int:
    """Count characters (excluding whitespace)"""
    return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))


def count_bytes(text: str) -> int:
    """Count UTF-8 bytes"""
    return len(text.encode('utf-8'))


def analyze_tokenizer(tokenizer_name: str, tokenizer, texts: List[str]) -> Dict:
    """
    Analyze a tokenizer on a corpus of texts
    Returns statistics about tokenization efficiency
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {tokenizer_name}...")
    print(f"{'='*50}")
    
    all_token_ids = []
    tokens_per_doc = []
    tokens_per_word = []
    tokens_per_char = []
    tokens_per_byte = []
    
    for text in texts:
        # Tokenize
        if hasattr(tokenizer, 'encode'):
            # Custom tokenizers (BPE, ByT5, Canine)
            token_ids = tokenizer.encode(text)
        else:
            # HuggingFace tokenizers (BERT, RoBERTa)
            token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        num_tokens = len(token_ids)
        num_words = count_words(text)
        num_chars = count_characters(text)
        num_bytes = count_bytes(text)
        
        # Store token IDs for vocabulary analysis
        all_token_ids.extend(token_ids)
        
        # Calculate ratios
        tokens_per_doc.append(num_tokens)
        if num_words > 0:
            tokens_per_word.append(num_tokens / num_words)
        if num_chars > 0:
            tokens_per_char.append(num_tokens / num_chars)
        if num_bytes > 0:
            tokens_per_byte.append(num_tokens / num_bytes)
    
    # Vocabulary statistics
    unique_tokens = len(set(all_token_ids))
    total_tokens = len(all_token_ids)
    token_counts = Counter(all_token_ids)
    most_common_tokens = token_counts.most_common(10)
    
    # Get vocabulary size
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'vocab_size'):
        vocab_size = tokenizer.tokenizer.vocab_size
    elif tokenizer_name == 'BPE':
        # Custom BPE
        vocab = tokenizer.build_vocab()
        vocab_size = len(vocab)
    else:
        vocab_size = "Unknown"
    
    # Calculate statistics
    stats = {
        'tokenizer': tokenizer_name,
        'vocab_size': vocab_size,
        'unique_tokens_used': unique_tokens,
        'total_tokens': total_tokens,
        'vocab_utilization': f"{(unique_tokens / vocab_size * 100):.2f}%" if isinstance(vocab_size, int) else "N/A",
        
        # Tokens per document
        'avg_tokens_per_doc': statistics.mean(tokens_per_doc),
        'median_tokens_per_doc': statistics.median(tokens_per_doc),
        'min_tokens_per_doc': min(tokens_per_doc),
        'max_tokens_per_doc': max(tokens_per_doc),
        
        # Compression efficiency
        'avg_tokens_per_word': statistics.mean(tokens_per_word),
        'avg_tokens_per_char': statistics.mean(tokens_per_char),
        'avg_tokens_per_byte': statistics.mean(tokens_per_byte),
        
        # Compression ratio (lower is better)
        'compression_ratio': statistics.mean(tokens_per_byte),  # tokens/byte
        
        # Most common tokens
        'most_common_tokens': most_common_tokens[:5]
    }
    
    return stats


def print_stats(stats: Dict):
    """Pretty print tokenization statistics"""
    print(f"\n{stats['tokenizer']} Statistics:")
    print(f"  Vocabulary Size: {stats['vocab_size']}")
    print(f"  Unique Tokens Used: {stats['unique_tokens_used']}")
    print(f"  Vocabulary Utilization: {stats['vocab_utilization']}")
    print(f"\n  Tokens per Document:")
    print(f"    Average: {stats['avg_tokens_per_doc']:.2f}")
    print(f"    Median: {stats['median_tokens_per_doc']:.2f}")
    print(f"    Range: {stats['min_tokens_per_doc']} - {stats['max_tokens_per_doc']}")
    print(f"\n  Compression Efficiency:")
    print(f"    Tokens per Word: {stats['avg_tokens_per_word']:.3f}")
    print(f"    Tokens per Char: {stats['avg_tokens_per_char']:.3f}")
    print(f"    Tokens per Byte: {stats['avg_tokens_per_byte']:.3f}")
    print(f"    Compression Ratio: {stats['compression_ratio']:.3f} (lower = better)")


def save_comparison_table(all_stats: List[Dict], output_path: str):
    """Save comparison table to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'tokenizer', 'vocab_size', 'unique_tokens_used', 'vocab_utilization',
            'avg_tokens_per_doc', 'median_tokens_per_doc', 
            'avg_tokens_per_word', 'avg_tokens_per_char', 'avg_tokens_per_byte',
            'compression_ratio'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for stats in all_stats:
            row = {k: v for k, v in stats.items() if k in fieldnames}
            # Format floats
            for key in ['avg_tokens_per_doc', 'median_tokens_per_doc', 
                       'avg_tokens_per_word', 'avg_tokens_per_char', 
                       'avg_tokens_per_byte', 'compression_ratio']:
                if key in row:
                    row[key] = f"{row[key]:.3f}"
            writer.writerow(row)
    
    print(f"\n✅ Comparison table saved to {output_path}")


def print_comparison_table(all_stats: List[Dict]):
    """Print a formatted comparison table"""
    print("\n" + "="*80)
    print("TOKENIZATION COMPARISON TABLE")
    print("="*80)
    
    # Header
    print(f"\n{'Tokenizer':<15} {'Vocab Size':<12} {'Tokens/Word':<12} {'Tokens/Doc':<12} {'Compression':<12}")
    print("-" * 80)
    
    # Rows
    for stats in all_stats:
        vocab_size = stats['vocab_size'] if isinstance(stats['vocab_size'], str) else f"{stats['vocab_size']:,}"
        print(f"{stats['tokenizer']:<15} {vocab_size:<12} "
              f"{stats['avg_tokens_per_word']:<12.3f} "
              f"{stats['avg_tokens_per_doc']:<12.1f} "
              f"{stats['compression_ratio']:<12.3f}")
    
    print("\n" + "="*80)
    print("Notes:")
    print("  - Compression Ratio = Tokens/Byte (lower is better)")
    print("  - Tokens/Word shows how many tokens needed per English word")
    print("  - Tokens/Doc shows average sequence length")
    print("="*80)


def analyze_special_cases(texts: List[str]):
    """Analyze how tokenizers handle special cases"""
    print("\n" + "="*80)
    print("SPECIAL CASE ANALYSIS")
    print("="*80)
    
    # Test cases
    test_cases = [
        ("Standard English", "The quick brown fox jumps over the lazy dog."),
        ("Numbers", "The year 2024 has 365 days and 12 months."),
        ("Mixed Case", "iPhone, JavaScript, and HTML5 are popular."),
        ("Punctuation", "Hello! How are you? I'm fine, thanks."),
        ("Unicode", "Café, naïve, résumé, 日本語"),
        ("Code", "def hello_world(): return 'Hello, World!'"),
        ("URL", "https://www.example.com/path?query=value"),
        ("Email", "user.name+tag@example.com"),
    ]
    
    # Initialize tokenizers
    tokenizers = {
        'BPE': CustomBPETokenizer(),
        'Canine': CanineTokenizer(),
        'BERT': AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'),
        'RoBERTa': AutoTokenizer.from_pretrained('roberta-base'),
    }
    
    # Load BPE tokenizer
    bpe_path = os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json')
    if os.path.exists(bpe_path):
        tokenizers['BPE'].load(bpe_path)
    else:
        print("⚠️  BPE tokenizer not found, skipping BPE analysis")
        del tokenizers['BPE']
    
    print(f"\n{'Test Case':<20} {'Text':<40} {'BPE':<8} {'Canine':<8} {'BERT':<8} {'RoBERTa':<8}")
    print("-" * 100)
    
    for case_name, text in test_cases:
        token_counts = {}
        for tok_name, tokenizer in tokenizers.items():
            try:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
                else:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                token_counts[tok_name] = len(tokens)
            except Exception as e:
                token_counts[tok_name] = "ERROR"
        
        # Truncate text for display
        display_text = text[:40] + "..." if len(text) > 40 else text
        
        print(f"{case_name:<20} {display_text:<40} "
              f"{str(token_counts.get('BPE', '-')):<8} "
              f"{str(token_counts.get('Canine', '-')):<8} "
              f"{str(token_counts.get('BERT', '-')):<8} "
              f"{str(token_counts.get('RoBERTa', '-')):<8}")


def main():
    """Main analysis pipeline"""
    print("🔍 Starting Tokenization Analysis")
    print(f"Analyzing documents from: {CORPUS_PATH}")
    print(f"Sample size: {MAX_SAMPLES} documents")
    
    # Load documents
    texts = load_documents(CORPUS_PATH, max_docs=MAX_SAMPLES)
    print(f"✅ Loaded {len(texts)} documents")
    
    # Initialize tokenizers
    tokenizers = {}
    
    # BPE
    bpe_path = os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json')
    if os.path.exists(bpe_path):
        print("\n📦 Loading BPE tokenizer...")
        tokenizers['BPE'] = CustomBPETokenizer()
        tokenizers['BPE'].load(bpe_path)
    else:
        print("⚠️  BPE tokenizer not found, skipping BPE analysis")
    
    # Canine
    print("📦 Loading Canine tokenizer...")
    tokenizers['Canine'] = CanineTokenizer()
    
    # BERT
    print("📦 Loading BERT tokenizer...")
    tokenizers['BERT-MiniLM'] = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # RoBERTa (BPE-based)
    print("📦 Loading RoBERTa tokenizer...")
    tokenizers['RoBERTa'] = AutoTokenizer.from_pretrained('roberta-base')
    
    # Analyze each tokenizer
    all_stats = []
    for name, tokenizer in tokenizers.items():
        stats = analyze_tokenizer(name, tokenizer, texts)
        print_stats(stats)
        all_stats.append(stats)
    
    # Print comparison table
    print_comparison_table(all_stats)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        os.path.dirname(__file__), 
        f"tokenization_analysis_{timestamp}.csv"
    )
    save_comparison_table(all_stats, output_path)
    
    # Analyze special cases
    analyze_special_cases(texts)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
