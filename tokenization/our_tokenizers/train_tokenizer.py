from datasets import load_dataset
import os
import json
from tokenization.our_tokenizers.BPE.BPE_tokenization import CustomBPETokenizer

# ------------------------------------------------------------
# STEP 1 — Load YOUR data (Natural Questions corpus)
# ------------------------------------------------------------

def load_nq_corpus(corpus_path="../../data_filtered/corpus_filtered.jsonl"):
    """Load NQ corpus from JSONL file - streaming for memory efficiency"""
    texts = []
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        return texts
    
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                doc = json.loads(line.strip())
                # Combine title and text - prioritize text over title
                title = doc.get('title', '').strip()
                text = doc.get('text', '').strip()
                
                # Use text if available, otherwise title, combine if both exist
                if text and title:
                    combined = f"{title} {text}"
                elif text:
                    combined = text
                elif title:
                    combined = title
                else:
                    continue
                
                # Clean whitespace
                combined = " ".join(combined.split())
                if combined:
                    texts.append(combined)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i}")
                continue
    
    print(f"✅ Loaded {len(texts)} documents")
    return texts


# ------------------------------------------------------------
# STEP 2 — Train BPE tokenizer on YOUR data (EFFICIENT)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Load YOUR Natural Questions data
    texts = load_nq_corpus()
    
    if not texts:
        print("❌ No texts loaded. Exiting.")
        exit(1)
    
    # Combine all texts - memory efficient way
    print("Combining texts...")
    corpus_text = "\n".join(texts)
    del texts  # Free memory immediately
    
    corpus_bytes = len(corpus_text.encode("utf-8"))
    corpus_size_mb = corpus_bytes / 1e6
    print(f"✅ Corpus size: {corpus_size_mb:.2f} MB ({corpus_bytes:,} bytes)")
    
    # Train tokenizer
    print("\n📚 Training BPE tokenizer on YOUR NQ data...")
    print("   This may take a minute or two...\n")
    tokenizer = CustomBPETokenizer()
    tokenizer.train(corpus_text, vocab_size=2000)
    
    # Free corpus memory
    del corpus_text
    
    # Save tokenizer
    tokenizer.save("bpe_tokenizer.json")
    print("\n✅ Tokenizer saved to bpe_tokenizer.json")
    
    # Print stats
    vocab = tokenizer.build_vocab()
    print(f"✅ Vocab size: {len(vocab):,} tokens")
    print(f"✅ Number of learned merges: {len(tokenizer.merges):,}")
    print(f"\n🎉 Your tokenizer is now optimized for NQ data!")
