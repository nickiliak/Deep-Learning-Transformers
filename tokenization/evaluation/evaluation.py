"""
Minimal Evaluation Script - MVP
Compares ByT5, Canine, and BPE tokenizers using Recall@K
"""

import json
import os
import sys
from typing import List, Dict, Tuple
from tqdm import tqdm
import csv
from datetime import datetime

# Database imports
from sqlmodel import Session, create_engine, select, text
from pgvector.sqlalchemy import Vector

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tokenization.our_tokenizers.ByT5.ByT5_embedding import ByT5Embedder
from tokenization.our_tokenizers.Canine.Canine_embedding import CanineEmbedder
from tokenization.our_tokenizers.BPE.BPE_embedding import BPEEmbedder


# Configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://nick:secret@localhost:5433/vectordb",
)
QUERIES_PATH = os.path.join(repo_root, "data_filtered", "queries_filtered.jsonl")

# Model configurations
MODELS = [
    {
        'name': 'ByT5',
        'embedder_class': ByT5Embedder,
        'model_id': 'google/byt5-small',
        'table_name': 'byt5_small',
        'vector_dim': 1472,
        'is_bpe': False
    },
    {
        'name': 'Canine',
        'embedder_class': CanineEmbedder,
        'model_id': 'google/canine-s',
        'table_name': 'canine_s',
        'vector_dim': 768,
        'is_bpe': False
    },
    {
        'name': 'BPE',
        'embedder_class': BPEEmbedder,
        'bpe_model_path': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'BPE',
        'vector_dim': 768,  # BPE uses d_model=768
        'is_bpe': True
    }
]


def load_queries(path: str) -> List[Dict]:
    """Load queries from JSONL file"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def split_queries(queries: List[Dict], val_ratio: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
    """
    Split queries into validation and test sets
    Simple split: first 50% validation, last 50% test
    """
    split_idx = int(len(queries) * val_ratio)
    return queries[:split_idx], queries[split_idx:]


def retrieve_top_k(query_embedding: List[float], table_name: str, k: int, engine) -> List[str]:
    """
    Retrieve top-k document IDs using pgvector similarity search
    Returns list of document IDs
    """
    with Session(engine) as session:
        # Use raw SQL for pgvector similarity search
        # Cast the parameter to vector type explicitly to avoid type mismatch
        # Quote table name to handle case-sensitive names like "BPE"
        query = text(f"""
            SELECT id, embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM "{table_name}"
            ORDER BY distance
            LIMIT :k
        """)
        
        result = session.execute(
            query,
            {"query_embedding": str(query_embedding), "k": k}
        )
        
        doc_ids = [row[0] for row in result]
        return doc_ids


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K
    Recall@K = (# relevant docs in top-K) / (total # relevant docs)
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(retrieved_set & relevant_set)
    recall = hits / len(relevant_set)
    
    return recall


def evaluate_model(model_config: Dict, queries: List[Dict], k_values: List[int], engine) -> Dict:
    """
    Evaluate a single model on queries
    Returns dict with recall scores for each K
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model_config['name']}...")
    print(f"{'='*50}")
    
    # Initialize embedder - special handling for BPE
    try:
        if model_config.get('is_bpe', False):
            # BPE uses bpe_model_path parameter
            embedder = model_config['embedder_class'](
                bpe_model_path=model_config['bpe_model_path']
            )
        else:
            # ByT5 and Canine use model_id parameter
            embedder = model_config['embedder_class'](model_config['model_id'])
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return None
    
    # Store recall scores for each K
    recall_scores = {k: [] for k in k_values}
    
    # Evaluate each query
    for query_obj in tqdm(queries, desc=f"{model_config['name']} queries"):
        query_text = query_obj['text']
        relevant_ids = query_obj['corpus-id']
        
        # Generate query embedding
        try:
            query_embedding = embedder.generate_embedding(query_text)
        except Exception as e:
            print(f"⚠️  Failed to embed query {query_obj['_id']}: {e}")
            continue
        
        # Retrieve documents
        try:
            retrieved_ids = retrieve_top_k(
                query_embedding, 
                model_config['table_name'], 
                max(k_values),  # Retrieve max K once
                engine
            )
        except Exception as e:
            print(f"⚠️  Failed to retrieve for query {query_obj['_id']}: {e}")
            continue
        
        # Calculate recall for each K
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            recall_scores[k].append(recall)
    
    # Calculate average recall for each K
    avg_recall = {}
    for k in k_values:
        if recall_scores[k]:
            avg_recall[k] = sum(recall_scores[k]) / len(recall_scores[k])
        else:
            avg_recall[k] = 0.0
    
    return {
        'model': model_config['name'],
        'recall_scores': avg_recall,
        'num_queries': len(queries)
    }


def save_results(results: List[Dict], output_path: str):
    """Save evaluation results to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Get all K values from first result
        k_values = list(results[0]['recall_scores'].keys()) if results else []
        
        fieldnames = ['model', 'num_queries'] + [f'recall@{k}' for k in k_values]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            row = {
                'model': result['model'],
                'num_queries': result['num_queries']
            }
            for k in k_values:
                row[f'recall@{k}'] = f"{result['recall_scores'][k]:.4f}"
            writer.writerow(row)
    
    print(f"\n✅ Results saved to {output_path}")


def print_results(results: List[Dict]):
    """Print results in a nice table format"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Queries evaluated: {result['num_queries']}")
        for k, score in result['recall_scores'].items():
            print(f"  Recall@{k}: {score:.4f}")


def main():
    """Main evaluation pipeline"""
    print("🚀 Starting Minimal Evaluation Pipeline (MVP)")
    print(f"Loading queries from: {QUERIES_PATH}")
    
    # Load queries
    all_queries = load_queries(QUERIES_PATH)
    print(f"✅ Loaded {len(all_queries)} queries")
    
    # Split into validation and test
    val_queries, test_queries = split_queries(all_queries, val_ratio=0.5)
    print(f"📊 Split: {len(val_queries)} validation, {len(test_queries)} test queries")
    
    # Use test set for evaluation (validation set available for future tuning)
    eval_queries = test_queries
    print(f"\n🎯 Evaluating on {len(eval_queries)} test queries")
    
    # K values to evaluate
    k_values = [1, 5, 10]
    print(f"📏 K values: {k_values}")
    
    # Connect to database
    engine = create_engine(DATABASE_URL)
    print(f"✅ Connected to database")
    
    # Evaluate each model
    results = []
    for model_config in MODELS:
        try:
            result = evaluate_model(model_config, eval_queries, k_values, engine)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Failed to evaluate {model_config['name']}: {e}")
    
    # Print and save results
    if results:
        print_results(results)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__), 
            f"results_{timestamp}.csv"
        )
        save_results(results, output_path)
    else:
        print("\n❌ No results to report. Make sure embeddings are in the database.")
        print("   Run pipeline.ipynb for each model first!")


if __name__ == "__main__":
    main()
