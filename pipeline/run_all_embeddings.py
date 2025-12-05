"""
Run All Embeddings Pipeline
Processes corpus with multiple embedding models and stores in PostgreSQL + pgvector

Usage:
    python run_all_embeddings.py --all
    python run_all_embeddings.py --bpe-lstm --bpe-transformer --bert
    python run_all_embeddings.py --skip-byt5 --skip-canine
"""

import sys
import os

# Set PyTorch memory allocation to reduce fragmentation BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
from typing import Type, List
from dotenv import load_dotenv
import torch
import platform
import json
from tqdm import tqdm

# Add parent directory to path (repo root from pipeline folder)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import embedders
from tokenization.our_tokenizers.ByT5.ByT5_embedding import ByT5Embedder
from tokenization.our_tokenizers.Canine.Canine_embedding import CanineEmbedder
from tokenization.our_tokenizers.BPE.BPE_LSTM_embedding import BPELSTMEmbedder
from tokenization.our_tokenizers.BPE.BPE_transformer_embedding import BPETransformerEmbedder
from tokenization.our_tokenizers.BPE.BPEpre import BPEPretrainedEmbedder
from tokenization.baseline.BERT.bert_embeddings import BertEmbedder

# Database imports
from sqlmodel import SQLModel, Field, Session, create_engine, Column, text
from pgvector.sqlalchemy import Vector


# Configuration
load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://nick:secret@localhost:5433/vectordb",
)

DATASET_PATH = os.path.join(repo_root, "data_filtered", "corpus_filtered.jsonl")


# Model configurations
MODELS = {
    'byt5': {
        'name': 'ByT5',
        'embedder_class': ByT5Embedder,
        'model_id': 'google/byt5-small',
        'table_name': 'byt5_small',
        'vector_dim': 1472,
        'batch_size': 64,  # Smaller batch for large model
    },
    'canine': {
        'name': 'Canine',
        'embedder_class': CanineEmbedder,
        'model_id': 'google/canine-s',
        'table_name': 'canine_s',
        'vector_dim': 768,
        'batch_size': 64,
    },
    'bpe-lstm': {
        'name': 'BPE-LSTM-Trained',
        'embedder_class': BPELSTMEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-lstm',
        'vector_dim': 256,
        'batch_size': 64,
    },
    'bpe-transformer': {
        'name': 'BPE-Transformer-Trained',
        'embedder_class': BPETransformerEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-transformer',
        'vector_dim': 256,
        'batch_size': 64,
    },
    'bpe-pretrained': {
        'name': 'BPE-Pretrained',
        'embedder_class': BPEPretrainedEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-pretrained',
        'vector_dim': 256,
        'batch_size': 64,
    },
    'bert': {
        'name': 'BERT-MiniLM',
        'embedder_class': BertEmbedder,
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'table_name': 'bert_minilm',
        'vector_dim': 384,
        'batch_size': 64,
    }
}


def create_table_class(table_name: str, dim: int) -> Type[SQLModel]:
    """
    Dynamically creates a SQLModel class for storing embeddings
    """
    class DynamicTable(SQLModel, table=True):
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}

        id: str = Field(primary_key=True)
        title: str
        text: str
        embedding: List[float] = Field(sa_column=Column(Vector(dim)))

    return DynamicTable


def check_device():
    """Check and display available compute device"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"OS: {platform.system()} {platform.machine()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Running on MPS (Metal Performance Shaders) - M3 GPU Activated!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Running on CPU - Embeddings will be slow")

    return device


def run_pipeline_for_model(model_config: dict, clear_existing: bool = True):
    """
    Run embedding pipeline for a single model
    
    Args:
        model_config: Model configuration dict
        clear_existing: Whether to drop existing table before creating new one
    """
    print(f"\n{'='*60}")
    print(f"Processing: {model_config['name']}")
    print(f"{'='*60}")
    
    # Setup Database
    engine = create_engine(DATABASE_URL)
    
    # Ensure pgvector extension exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Clear existing table if requested
    if clear_existing:
        print(f"--- Clearing existing table: {model_config['table_name']} ---")
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{model_config["table_name"]}"'))
            conn.commit()

    # Define the Table Model
    TableClass = create_table_class(model_config['table_name'], model_config['vector_dim'])
    SQLModel.metadata.create_all(engine)

    # Initialize ML Model
    try:
        embedder = model_config['embedder_class'](model_config['model_id'])
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return False

    # Process JSONL and Insert
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return False

    print(f"--- Dataset: {DATASET_PATH} ---")
    print(f"--- Table: {model_config['table_name']} ---")
    print(f"--- Batch size: {model_config['batch_size']} ---")

    data_buffer = []
    text_buffer = []
    metadata_buffer = []
    BATCH_SIZE = 100 

    with Session(engine) as session:
        with open(DATASET_PATH, mode='r', encoding='utf-8') as f:
            
            for line in tqdm(f, desc=f"Embedding with {model_config['name']}"):
                try:
                    if not line.strip():
                        continue

                    # Parse JSON
                    row = json.loads(line)

                    # Extract Data
                    doc_id = row.get('_id')
                    title = row.get('title', '')
                    doc_text = row.get('text', '')

                    if not doc_id:
                        continue

                    # Prepare text and metadata for batch embedding
                    full_content = f"{title}: {doc_text}"
                    text_buffer.append(full_content)
                    metadata_buffer.append({'id': doc_id, 'title': title, 'text': doc_text})

                    # Process batch when buffer is full
                    if len(text_buffer) >= model_config['batch_size']:
                        if hasattr(embedder, 'generate_embeddings_batch'):
                            vectors = embedder.generate_embeddings_batch(text_buffer)
                        else:
                            vectors = [embedder.generate_embedding(text) for text in text_buffer]
                        
                        # Create records
                        for meta, vector in zip(metadata_buffer, vectors):
                            record = TableClass(
                                id=meta['id'],
                                title=meta['title'],
                                text=meta['text'],
                                embedding=vector
                            )
                            data_buffer.append(record)
                        
                        # Clear buffers
                        text_buffer = []
                        metadata_buffer = []

                    # Batch Commit to DB
                    if len(data_buffer) >= BATCH_SIZE:
                        session.add_all(data_buffer)
                        session.commit()
                        data_buffer = []

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error processing doc: {e}")

            # Process remaining texts
            if text_buffer:
                if hasattr(embedder, 'generate_embeddings_batch'):
                    vectors = embedder.generate_embeddings_batch(text_buffer)
                else:
                    vectors = [embedder.generate_embedding(text) for text in text_buffer]
                    
                for meta, vector in zip(metadata_buffer, vectors):
                    record = TableClass(
                        id=meta['id'],
                        title=meta['title'],
                        text=meta['text'],
                        embedding=vector
                    )
                    data_buffer.append(record)

            # Commit remaining records
            if data_buffer:
                session.add_all(data_buffer)
                session.commit()

    print(f"✅ {model_config['name']} completed successfully!")
    
    # Memory cleanup: explicitly delete embedder and clear GPU cache
    print(f"--- Cleaning up memory for {model_config['name']} ---")
    
    # Close session and engine to release database connections
    session.close()
    engine.dispose()
    
    # Delete embedder and related objects
    del embedder
    del engine
    del session
    del TableClass
    
    # Force Python garbage collection
    import gc
    gc.collect()
    
    # Clear PyTorch CUDA cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Synchronize GPU to ensure cache is actually cleared
        torch.cuda.synchronize()
        print(f"    GPU cache cleared and synchronized")
    
    print(f"    Memory cleanup complete\n")
    
    return True


def run_embeddings_pipeline(
    models=None,
    clear_existing=True,
):
    """
    Run embedding pipeline programmatically (without argparse)
    
    Args:
        models: List of models to run. If None, runs all models.
                Options: 'byt5', 'canine', 'bpe-lstm', 'bpe-transformer', 'bert'
        clear_existing: Whether to clear existing tables before inserting
    """
    # Check device
    check_device()
    
    # Default to all models if not specified
    if models is None:
        models = ['byt5', 'canine', 'bpe-lstm', 'bpe-transformer', 'bert']
    
    # Ensure models is a list
    if isinstance(models, str):
        models = [models]
    
    # Validate model names
    valid_models = ['byt5', 'canine', 'bpe-lstm', 'bpe-transformer', 'bert']
    models_to_run = [m for m in models if m in valid_models]
    
    if not models_to_run:
        print(f"⚠️  No valid models specified. Valid options: {', '.join(valid_models)}")
        return {}
    
    print(f"\n🚀 Running pipeline for {len(models_to_run)} model(s): {', '.join(models_to_run)}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Database: {DATABASE_URL}")
    print(f"Clear existing tables: {clear_existing}\n")
    
    # Run pipeline for each model
    results = {}
    for model_key in models_to_run:
        success = run_pipeline_for_model(MODELS[model_key], clear_existing=clear_existing)
        results[model_key] = success
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for model_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{MODELS[model_key]['name']}: {status}")
    
    print("\n🎉 All pipelines completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run embedding pipeline for multiple models')
    
    # Options to run specific models
    parser.add_argument('--all', action='store_true', help='Run all models')
    parser.add_argument('--byt5', action='store_true', help='Run ByT5')
    parser.add_argument('--canine', action='store_true', help='Run Canine')
    parser.add_argument('--bpe-lstm', action='store_true', help='Run BPE-LSTM')
    parser.add_argument('--bpe-transformer', action='store_true', help='Run BPE-Transformer')
    parser.add_argument('--bert', action='store_true', help='Run BERT-MiniLM')
    
    # Options to skip specific models (useful with --all)
    parser.add_argument('--skip-byt5', action='store_true', help='Skip ByT5')
    parser.add_argument('--skip-canine', action='store_true', help='Skip Canine')
    parser.add_argument('--skip-bpe-lstm', action='store_true', help='Skip BPE-LSTM')
    parser.add_argument('--skip-bpe-transformer', action='store_true', help='Skip BPE-Transformer')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT-MiniLM')
    
    # Other options
    parser.add_argument('--no-clear', action='store_true', help='Do not clear existing tables (append mode)')
    
    args = parser.parse_args()
    
    # Check device
    check_device()
    
    # Determine which models to run
    models_to_run = []
    
    if args.all:
        # Run all models unless explicitly skipped
        if not args.skip_byt5:
            models_to_run.append('byt5')
        if not args.skip_canine:
            models_to_run.append('canine')
        if not args.skip_bpe_lstm:
            models_to_run.append('bpe-lstm')
        if not args.skip_bpe_transformer:
            models_to_run.append('bpe-transformer')
        if not args.skip_bert:
            models_to_run.append('bert')
    else:
        # Run only explicitly requested models
        if args.byt5:
            models_to_run.append('byt5')
        if args.canine:
            models_to_run.append('canine')
        if args.bpe_lstm:
            models_to_run.append('bpe-lstm')
        if args.bpe_transformer:
            models_to_run.append('bpe-transformer')
        if args.bert:
            models_to_run.append('bert')
    
    # If no models specified, show help
    if not models_to_run:
        parser.print_help()
        print("\n⚠️  No models specified. Use --all or specify individual models.")
        return
    
    print(f"\n🚀 Running pipeline for {len(models_to_run)} model(s): {', '.join(models_to_run)}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Database: {DATABASE_URL}")
    print(f"Clear existing tables: {not args.no_clear}\n")
    
    # Run pipeline for each model
    results = {}
    for model_key in models_to_run:
        success = run_pipeline_for_model(MODELS[model_key], clear_existing=not args.no_clear)
        results[model_key] = success
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for model_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{MODELS[model_key]['name']}: {status}")
    
    print("\n🎉 All pipelines completed!")


if __name__ == "__main__":
    main()
