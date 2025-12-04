"""
Run All Embeddings Pipeline - CSV Version
Processes corpus with multiple embedding models and stores in CSV files.

Usage:
    python run_all_embeddings_csv.py --all
    python run_all_embeddings_csv.py --bpe-lstm --bert
"""

import sys
import os
import argparse
import json
import csv
from typing import List
from dotenv import load_dotenv
import torch
import platform
from tqdm import tqdm

# Add parent directory to path (repo root from pipeline folder)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import embedders
# (Ensure these paths remain valid in your project structure)
from tokenization.our_tokenizers.ByT5.ByT5_embedding import ByT5Embedder
from tokenization.our_tokenizers.Canine.Canine_embedding import CanineEmbedder
from tokenization.our_tokenizers.BPE.BPE_LSTM_embedding import BPELSTMEmbedder
from tokenization.our_tokenizers.BPE.BPE_transformer_embedding import BPETransformerEmbedder
from tokenization.baseline.BERT.bert_embeddings import BertEmbedder

# Configuration
load_dotenv()

DATASET_PATH = os.path.join(repo_root, "data_filtered", "corpus_filtered.jsonl")
OUTPUT_DIR = os.path.join(repo_root, "embeddings_processed")

# Model configurations
MODELS = {
    'byt5': {
        'name': 'ByT5',
        'embedder_class': ByT5Embedder,
        'model_id': 'google/byt5-small',
        'table_name': 'byt5_small', # This will be used as the filename
        'batch_size': 8,
    },
    'canine': {
        'name': 'Canine',
        'embedder_class': CanineEmbedder,
        'model_id': 'google/canine-s',
        'table_name': 'canine_s',
        'batch_size': 16,
    },
    'bpe-lstm': {
        'name': 'BPE-LSTM-Trained',
        'embedder_class': BPELSTMEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-lstm',
        'batch_size': 16,
    },
    'bpe-transformer': {
        'name': 'BPE-Transformer-Trained',
        'embedder_class': BPETransformerEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-transformer',
        'batch_size': 16,
    },
    'bert': {
        'name': 'BERT-MiniLM',
        'embedder_class': BertEmbedder,
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'table_name': 'bert_minilm',
        'batch_size': 32,
    }
}

def check_device():
    """Check and display available compute device"""
    print(f"PyTorch version: {torch.__version__}")
    
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


def run_pipeline_for_model(model_config: dict):
    """
    Run embedding pipeline for a single model and save to CSV
    """
    print(f"\n{'='*60}")
    print(f"Processing: {model_config['name']}")
    print(f"{'='*60}")
    
    # 1. Prepare output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. Define CSV filename
    filename = f"{model_config['table_name']}.csv"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    # 3. Initialize ML Model
    try:
        embedder = model_config['embedder_class'](model_config['model_id'])
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return False

    # 4. Process JSONL and Write to CSV
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return False

    print(f"--- Dataset: {DATASET_PATH} ---")
    print(f"--- Output: {file_path} ---")

    text_buffer = []
    metadata_buffer = []
    
    # Open CSV file in write mode ('w')
    # newline='' is important to avoid extra blank lines on Windows
    with open(file_path, mode='w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        header = ['id', 'title', 'text', 'embedding']
        csv_writer.writerow(header)

        with open(DATASET_PATH, mode='r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Embedding with {model_config['name']}"):
                try:
                    if not line.strip():
                        continue

                    # Parse JSON
                    row = json.loads(line)
                    doc_id = row.get('_id')
                    title = row.get('title', '')
                    doc_text = row.get('text', '')

                    if not doc_id:
                        continue

                    full_content = f"{title}: {doc_text}"
                    text_buffer.append(full_content)
                    metadata_buffer.append({'id': doc_id, 'title': title, 'text': doc_text})

                    # Process Batch
                    if len(text_buffer) >= model_config['batch_size']:
                        if hasattr(embedder, 'generate_embeddings_batch'):
                            vectors = embedder.generate_embeddings_batch(text_buffer)
                        else:
                            vectors = [embedder.generate_embedding(text) for text in text_buffer]
                        
                        # Write rows to CSV
                        rows_to_write = []
                        for meta, vector in zip(metadata_buffer, vectors):
                            # Convert vector (list of floats) to JSON string for storage in a single cell
                            vector_str = json.dumps(vector) 
                            rows_to_write.append([meta['id'], meta['title'], meta['text'], vector_str])
                        
                        csv_writer.writerows(rows_to_write)
                        
                        # Clear buffers
                        text_buffer = []
                        metadata_buffer = []

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error processing doc: {e}")

            # Process remaining items in buffer if any
            if text_buffer:
                if hasattr(embedder, 'generate_embeddings_batch'):
                    vectors = embedder.generate_embeddings_batch(text_buffer)
                else:
                    vectors = [embedder.generate_embedding(text) for text in text_buffer]
                
                rows_to_write = []
                for meta, vector in zip(metadata_buffer, vectors):
                    vector_str = json.dumps(vector)
                    rows_to_write.append([meta['id'], meta['title'], meta['text'], vector_str])
                
                csv_writer.writerows(rows_to_write)

    print(f"✅ {model_config['name']} completed successfully! Saved to {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run embedding pipeline to CSV')
    
    parser.add_argument('--all', action='store_true', help='Run all models')
    parser.add_argument('--byt5', action='store_true', help='Run ByT5')
    parser.add_argument('--canine', action='store_true', help='Run Canine')
    parser.add_argument('--bpe-lstm', action='store_true', help='Run BPE-LSTM')
    parser.add_argument('--bpe-transformer', action='store_true', help='Run BPE-Transformer')
    parser.add_argument('--bert', action='store_true', help='Run BERT-MiniLM')
    
    parser.add_argument('--skip-byt5', action='store_true', help='Skip ByT5')
    parser.add_argument('--skip-canine', action='store_true', help='Skip Canine')
    parser.add_argument('--skip-bpe-lstm', action='store_true', help='Skip BPE-LSTM')
    parser.add_argument('--skip-bpe-transformer', action='store_true', help='Skip BPE-Transformer')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT-MiniLM')
    
    args = parser.parse_args()
    
    check_device()
    
    models_to_run = []
    
    if args.all:
        if not args.skip_byt5: models_to_run.append('byt5')
        if not args.skip_canine: models_to_run.append('canine')
        if not args.skip_bpe_lstm: models_to_run.append('bpe-lstm')
        if not args.skip_bpe_transformer: models_to_run.append('bpe-transformer')
        if not args.skip_bert: models_to_run.append('bert')
    else:
        if args.byt5: models_to_run.append('byt5')
        if args.canine: models_to_run.append('canine')
        if args.bpe_lstm: models_to_run.append('bpe-lstm')
        if args.bpe_transformer: models_to_run.append('bpe-transformer')
        if args.bert: models_to_run.append('bert')
    
    if not models_to_run:
        parser.print_help()
        print("\n⚠️  No models specified. Use --all or specify individual models.")
        return
    
    print(f"\n🚀 Running CSV pipeline for {len(models_to_run)} model(s): {', '.join(models_to_run)}")
    
    results = {}
    for model_key in models_to_run:
        success = run_pipeline_for_model(MODELS[model_key])
        results[model_key] = success
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for model_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{MODELS[model_key]['name']}: {status}")
    
    print("\n🎉 All CSVs generated!")

if __name__ == "__main__":
    main()