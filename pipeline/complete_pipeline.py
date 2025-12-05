"""
Complete Pipeline: Preprocess -> Tokenizer -> Train Models -> Embeddings -> Evaluation
Single file orchestration with parametrized stages.
"""

import os
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# ============================================================================
# PIPELINE CONFIGURATION - Modify these parameters
# ============================================================================
@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    # Skip flags
    skip_preprocess: bool = True
    skip_tokenizer: bool = True
    skip_lstm: bool = False
    skip_transformer: bool = False
    skip_embeddings: bool = False
    skip_evaluation: bool = True
    
    # Preprocessing parameters
    corpus_size: int = 3000
    
    # Tokenizer parameters
    tokenizer_vocab_size: int = 2000
    
    # Model training parameters
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_seq_length: int = 128
    lstm_learning_rate: float = 0.001
    
    transformer_epochs: int = 50
    transformer_batch_size: int = 32
    transformer_seq_length: int = 128
    transformer_learning_rate: float = 0.001

# ============================================================================
# STAGE 1: PREPROCESSING
# ============================================================================
def stage_preprocess(config: PipelineConfig):
    """Preprocess NQ dataset: filter corpus and align queries"""
    if config.skip_preprocess:
        print("\n[SKIP] Preprocessing")
        return
    
    print("\n" + "="*80)
    print("STAGE 1: PREPROCESSING")
    print("="*80)
    
    from data_processing.nq_preprocess import preprocess_data
    
    try:
        print(f"\nParameters:")
        print(f"  Corpus size: {config.corpus_size}")
        
        corpus_file, queries_file = preprocess_data(corpus_size=config.corpus_size)
        print(f"\n[OK] Preprocessing complete")
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        raise


# ============================================================================
# STAGE 2: TOKENIZER TRAINING
# ============================================================================
def stage_tokenizer(config: PipelineConfig):
    """Train BPE tokenizer on C4 dataset"""
    if config.skip_tokenizer:
        print("\n[SKIP] Tokenizer training")
        return
    
    print("\n" + "="*80)
    print("STAGE 2: TOKENIZER TRAINING")
    print("="*80)
    
    # Change to tokenizer directory and run the script
    tokenizer_script = repo_root / 'tokenization' / 'our_tokenizers' / 'train_tokenizer.py'
    
    try:
        print(f"\nParameters:")
        print(f"  Vocab size: {config.tokenizer_vocab_size}")
        print(f"\nRunning tokenizer training from {tokenizer_script}...")
        result = subprocess.run(
            [sys.executable, str(tokenizer_script)],
            cwd=repo_root / 'tokenization' / 'our_tokenizers',
            check=True
        )
        print(f"\n[OK] Tokenizer training complete")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Tokenizer training failed with exit code {e.returncode}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Tokenizer training failed: {e}")
        raise


# ============================================================================
# STAGE 3A: TRAIN LSTM MODEL
# ============================================================================
def stage_train_lstm(config: PipelineConfig):
    """Train LSTM language model with BPE tokenization"""
    if config.skip_lstm:
        print("\n[SKIP] LSTM model training")
        return
    
    print("\n" + "="*80)
    print("STAGE 3A: LSTM MODEL TRAINING")
    print("="*80)
    
    from models.LSTM.training.train_bpe_lstm import main as train_lstm_main
    
    try:
        print(f"\nParameters:")
        print(f"  Epochs: {config.lstm_epochs}")
        print(f"  Batch size: {config.lstm_batch_size}")
        print(f"  Sequence length: {config.lstm_seq_length}")
        print(f"  Learning rate: {config.lstm_learning_rate}")
        
        train_lstm_main(
            batch_size=config.lstm_batch_size,
            seq_length=config.lstm_seq_length,
            num_epochs=config.lstm_epochs,
            learning_rate=config.lstm_learning_rate
        )
        print(f"\n[OK] LSTM training complete")
    except Exception as e:
        print(f"\n[ERROR] LSTM training failed: {e}")
        raise


# ============================================================================
# STAGE 3B: TRAIN TRANSFORMER MODEL
# ============================================================================
def stage_train_transformer(config: PipelineConfig):
    """Train Transformer language model with BPE tokenization"""
    if config.skip_transformer:
        print("\n[SKIP] Transformer model training")
        return
    
    print("\n" + "="*80)
    print("STAGE 3B: TRANSFORMER MODEL TRAINING")
    print("="*80)
    
    from models.Transformer.training.train_bpe_transformer import main as train_transformer_main
    
    try:
        print(f"\nParameters:")
        print(f"  Epochs: {config.transformer_epochs}")
        print(f"  Batch size: {config.transformer_batch_size}")
        print(f"  Sequence length: {config.transformer_seq_length}")
        print(f"  Learning rate: {config.transformer_learning_rate}")
        
        train_transformer_main(
            batch_size=config.transformer_batch_size,
            seq_length=config.transformer_seq_length,
            num_epochs=config.transformer_epochs,
            learning_rate=config.transformer_learning_rate
        )
        print(f"\n[OK] Transformer training complete")
    except Exception as e:
        print(f"\n[ERROR] Transformer training failed: {e}")
        raise


# ============================================================================
# STAGE 4: EMBEDDINGS GENERATION
# ============================================================================
def stage_embeddings(config: PipelineConfig):
    """Generate embeddings using all models and store in database"""
    if config.skip_embeddings:
        print("\n[SKIP] Embeddings generation")
        return
    
    print("\n" + "="*80)
    print("STAGE 4: EMBEDDINGS GENERATION")
    print("="*80)
    
    from pipeline.run_all_embeddings import run_embeddings_pipeline
    
    try:
        # Prepare models to run (all by default)
        models = ['byt5', 'canine', 'bpe-lstm', 'bpe-transformer', 'bert']
        
        print(f"\nParameters:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Clear tables: True")
        
        results = run_embeddings_pipeline(
            models=models,
            clear_existing=True
        )
        print(f"\n[OK] Embeddings generation complete")
    except Exception as e:
        print(f"\n[ERROR] Embeddings generation failed: {e}")
        raise


# ============================================================================
# STAGE 5: EVALUATION
# ============================================================================
def stage_evaluation(config: PipelineConfig):
    """Evaluate all embedding models on retrieval task"""
    if config.skip_evaluation:
        print("\n[SKIP] Evaluation")
        return
    
    print("\n" + "="*80)
    print("STAGE 5: EVALUATION")
    print("="*80)
    
    from tokenization.evaluation.evaluation import main as evaluation_main
    
    try:
        evaluation_main()
        print(f"\n[OK] Evaluation complete")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        raise


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
def main(config: PipelineConfig = None):
    """Run the complete pipeline with given configuration"""
    if config is None:
        config = PipelineConfig()
    
    print("\n" + "="*80)
    print("COMPLETE PIPELINE EXECUTION")
    print("="*80)
    
    try:
        # Run stages with config
        stage_preprocess(config)
        stage_tokenizer(config)
        stage_train_lstm(config)
        stage_train_transformer(config)
        stage_embeddings(config)
        stage_evaluation(config)
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print("\nConfiguration Summary:")
        print(f"  Skip preprocess: {config.skip_preprocess}")
        print(f"  Skip tokenizer: {config.skip_tokenizer}")
        print(f"  Skip LSTM: {config.skip_lstm}")
        print(f"  Skip Transformer: {config.skip_transformer}")
        print(f"  Skip embeddings: {config.skip_embeddings}")
        print(f"  Skip evaluation: {config.skip_evaluation}")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("[ERROR] PIPELINE FAILED")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURE YOUR PIPELINE HERE
    # =========================================================================
    config = PipelineConfig(
        skip_preprocess=True,
        skip_tokenizer=True,
        skip_lstm=True,
        skip_transformer=True,
        skip_embeddings=False,
        skip_evaluation=True,
        corpus_size=100,
        tokenizer_vocab_size=2000,
        lstm_epochs=1,
        lstm_batch_size=32,
        lstm_seq_length=128,
        lstm_learning_rate=0.001,
        transformer_epochs=1,
        transformer_batch_size=32,
        transformer_seq_length=128,
        transformer_learning_rate=0.001,
    )
    
    main(config)
