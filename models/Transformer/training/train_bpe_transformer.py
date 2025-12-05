"""
Train Transformer Language Model with BPE tokenization
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.Transformer.training.transformer_model import SimpleTransformer_LM
from models.LSTM.training.dataset import LanguageModelingDataset, load_documents_from_jsonl
from tokenization.our_tokenizers.BPE.BPE_tokenization import CustomBPETokenizer


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0) * input_ids.size(1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{torch.exp(loss):.2f}'
        })
    
    avg_loss = total_loss / len(train_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0) * input_ids.size(1)
    
    avg_loss = total_loss / len(val_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()


def calculate_bits_per_character(model, tokenizer, val_loader, device):
    """
    Calculate bits per character (fair metric across tokenizers)
    BPC = log2(perplexity per character)
    """
    model.eval()
    total_loss = 0
    total_chars = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
            
            # Rough estimate: assume avg 4 chars per token for BPE
            chars_in_batch = input_ids.numel() * 4
            total_loss += loss.item() * chars_in_batch
            total_chars += chars_in_batch
    
    bpc = (total_loss / total_chars) / torch.log(torch.tensor(2.0))
    return bpc.item()


def main(batch_size=32, seq_length=128, num_epochs=50, learning_rate=0.001):
    """
    Train Transformer Language Model with BPE Tokenization
    
    Args:
        batch_size: Batch size for training
        seq_length: Sequence length for language modeling
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    print("="*60)
    print("Training Transformer Language Model with BPE Tokenization")
    print("="*60)
    
    # Configuration
    CORPUS_PATH = os.path.join(repo_root, "data_filtered", "corpus_filtered.jsonl")
    BPE_MODEL_PATH = os.path.join(repo_root, "tokenization", "vocabularies", "bpe_tokenizer.json")
    
    BATCH_SIZE = batch_size
    SEQ_LENGTH = seq_length
    NUM_EPOCHS = num_epochs
    LEARNING_RATE = learning_rate
    MAX_DOCS = 5000  # Use same as LSTM for fair comparison
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Load BPE tokenizer
    print(f"\n📦 Loading BPE tokenizer from {BPE_MODEL_PATH}")
    tokenizer = CustomBPETokenizer()
    tokenizer.load(BPE_MODEL_PATH)
    vocab = tokenizer.build_vocab()
    vocab_size = len(vocab) + 1  # +1 for padding
    print(f"   Vocabulary size: {vocab_size}")
    
    # Load data
    print(f"\n📚 Loading documents from {CORPUS_PATH}")
    texts = load_documents_from_jsonl(CORPUS_PATH, max_docs=MAX_DOCS)
    print(f"   Loaded {len(texts)} documents")
    
    # Create dataset
    print(f"\n🔨 Creating dataset...")
    full_dataset = LanguageModelingDataset(tokenizer, texts, seq_length=SEQ_LENGTH, stride=64)
    
    # Split train/val/test (80/10/10)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val:   {len(val_dataset)} examples")
    print(f"   Test:  {len(test_dataset)} examples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🧠 Creating Transformer model...")
    model = SimpleTransformer_LM(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.2
    )
    model = model.to(device)
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(repo_root, "models", "Transformer", "transformer_bpe_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, save_path)
            print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
    
    # Final evaluation on test set
    print(f"\n📊 Final evaluation on test set...")
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    bpc = calculate_bits_per_character(model, tokenizer, test_loader, device)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    print(f"Bits per Char:   {bpc:.3f}")
    print(f"{'='*60}")
    
    # Save final model
    final_path = os.path.join(repo_root, "models", "Transformer", "transformer_bpe_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'bpc': bpc,
    }, final_path)
    print(f"\n✅ Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
