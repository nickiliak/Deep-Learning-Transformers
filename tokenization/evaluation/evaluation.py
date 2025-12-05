"""
Minimal Evaluation Script - MVP
Compares ByT5, Canine, and BPE tokenizers using Recall@K
"""

import json
import os
import sys
import time
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
from tokenization.our_tokenizers.BPE.BPE_LSTM_embedding import BPELSTMEmbedder
from tokenization.our_tokenizers.BPE.BPE_transformer_embedding import BPETransformerEmbedder
from tokenization.our_tokenizers.BPE.BPEpre import BPEPretrainedEmbedder
from tokenization.baseline.BERT.bert_embeddings import BertEmbedder

# Configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://nick:secret@localhost:5433/vectordb",
)
QUERIES_PATH = os.path.join(repo_root, "data_filtered", "queries_filtered.jsonl")

# Model configurations - MUST match table names created by run_all_embeddings.py
MODELS = [
    {
        'name': 'ByT5',
        'embedder_class': ByT5Embedder,
        'model_id': 'google/byt5-small',
        'table_name': 'byt5_small',
        'vector_dim': 1472,
        'is_bpe': False,
    },
    {
        'name': 'Canine',
        'embedder_class': CanineEmbedder,
        'model_id': 'google/canine-s',
        'table_name': 'canine_s',
        'vector_dim': 768,
        'is_bpe': False,
    },
    {
        'name': 'BPE-LSTM-Trained',
        'embedder_class': BPELSTMEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-lstm',
        'vector_dim': 256,
        'is_bpe': True,
    },
    {
        'name': 'BPE-Transformer-Trained',
        'embedder_class': BPETransformerEmbedder,
        'model_id': os.path.join(repo_root, 'tokenization', 'vocabularies', 'bpe_tokenizer.json'),
        'table_name': 'bpe-transformer',
        'vector_dim': 256,
        'is_bpe': True,
    },
    {
        'name': 'BPE-Pretrained',
        'embedder_class': BPEPretrainedEmbedder,
        'model_id': 'roberta-base',
        'table_name': 'bpe-pretrained',
        'vector_dim': 768,
        'is_bpe': False,
    },
    {
        'name': 'BERT-MiniLM',
        'embedder_class': BertEmbedder,
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'table_name': 'bert_minilm',
        'vector_dim': 384,
        'is_bpe': False,
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
    # Use engine.connect() instead of Session to get isolated connections
    # This prevents transaction abort issues
    with engine.connect() as connection:
        # Use raw SQL for pgvector similarity search
        # Try with exact case first, then lowercase if that fails
        # PostgreSQL stores unquoted names as lowercase, but SQLModel creates them with exact case
        try:
            # Try lowercase (PostgreSQL default)
            query = text(f"""
                SELECT id, embedding <=> CAST(:query_embedding AS vector) AS distance
                FROM {table_name.lower()}
                ORDER BY distance
                LIMIT :k
            """)
            
            result = connection.execute(
                query,
                {"query_embedding": str(query_embedding), "k": k}
            )
            
            doc_ids = [row[0] for row in result]
            return doc_ids
        except Exception as e:
            # If lowercase fails, rollback and try with exact case quoted
            connection.rollback()
            query = text(f"""
                SELECT id, embedding <=> CAST(:query_embedding AS vector) AS distance
                FROM "{table_name}"
                ORDER BY distance
                LIMIT :k
            """)
            
            result = connection.execute(
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


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank)
    MRR = 1 / rank of first relevant document
    Returns 0 if no relevant documents found
    """
    if not relevant_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    
    # Find position of first relevant document (1-indexed)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    # No relevant document found
    return 0.0


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Precision@K
    Precision@K = (# relevant docs in top-K) / K
    """
    if not relevant_ids or k == 0:
        return 0.0
    
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(retrieved_set & relevant_set)
    precision = hits / k
    
    return precision


def evaluate_model(model_config: Dict, queries: List[Dict], k_values: List[int], engine) -> Dict:
    """
    Evaluate a single model on queries using BATCH PROCESSING for embeddings.
    Returns dict with recall scores for each K
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model_config['name']}...")
    print(f"{'='*50}")
    
    # Initialize embedder - standard loading without explicit device handling
    try:
        # Check if it's a trained BPE model (uses file path)
        if model_config.get('is_bpe', False):
            # Trained BPE models use model_id as the bpe_model_path
            embedder = model_config['embedder_class'](
                bpe_model_path=model_config['model_id']
            )
        else:
            # All other models use model_id parameter directly
            embedder = model_config['embedder_class'](model_config['model_id'])
            embedder = model_config['embedder_class'](model_config['model_id'])
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return None
    
    # ---------------------------------------------------------
    # STEP 1: BATCH EMBEDDING GENERATION
    # ---------------------------------------------------------
    # Instead of processing 1 query -> 1 retrieval -> repeat,
    # we generate ALL embeddings first in batches to maximize GPU throughput.
    
    BATCH_SIZE = 64  # Process 64 queries at once
    all_query_texts = [q['text'] for q in queries]
    all_embeddings = [] # Store all calculated vectors here
    
    print(f"⚡ Generating embeddings in batches of {BATCH_SIZE}...")
    embed_start_total = time.time()
    
    # Process queries in chunks
    for i in range(0, len(all_query_texts), BATCH_SIZE):
        batch_texts = all_query_texts[i : i + BATCH_SIZE]
        
        try:
            # Check if the embedder supports batching (the BPE-LSTM one does)
            if hasattr(embedder, 'generate_embeddings_batch'):
                batch_vectors = embedder.generate_embeddings_batch(batch_texts)
            else:
                # Fallback for models that might not have the batch method yet
                batch_vectors = [embedder.generate_embedding(t) for t in batch_texts]
            
            all_embeddings.extend(batch_vectors)
            
        except Exception as e:
            print(f"⚠️ Error in batch {i}: {e}")
            # Append None to keep indices aligned if a batch fails
            all_embeddings.extend([None] * len(batch_texts))

    # Calculate average embedding time per query based on total batch time
    total_embed_time_ms = (time.time() - embed_start_total) * 1000
    avg_embedding_time = total_embed_time_ms / len(queries) if queries else 0

    # ---------------------------------------------------------
    # STEP 2: RETRIEVAL AND SCORING
    # ---------------------------------------------------------
    
    recall_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    mrr_scores = []
    retrieval_times = []
    
    # Iterate through queries, using the pre-calculated embeddings
    for i, query_obj in enumerate(tqdm(queries, desc=f"{model_config['name']} retrieval")):
        relevant_ids = query_obj['corpus-id']
        query_embedding = all_embeddings[i]
        
        # Skip if embedding failed
        if query_embedding is None:
            continue
        
        # Retrieve documents (Database Search)
        try:
            retrieval_start = time.time()
            retrieved_ids = retrieve_top_k(
                query_embedding, 
                model_config['table_name'], 
                max(k_values),
                engine
            )
            retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to ms
            retrieval_times.append(retrieval_time)
        except Exception as e:
            print(f"⚠️  Failed to retrieve for query {query_obj['_id']}: {e}")
            continue
        
        # Calculate Metrics (Recall, Precision, MRR)
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            recall_scores[k].append(recall)
            
            precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            precision_scores[k].append(precision)
        
        mrr = calculate_mrr(retrieved_ids, relevant_ids)
        mrr_scores.append(mrr)
    
    # ---------------------------------------------------------
    # STEP 3: AGGREGATE RESULTS
    # ---------------------------------------------------------
    
    # Calculate averages
    avg_recall = {}
    for k in k_values:
        avg_recall[k] = sum(recall_scores[k]) / len(recall_scores[k]) if recall_scores[k] else 0.0
    
    avg_precision = {}
    for k in k_values:
        avg_precision[k] = sum(precision_scores[k]) / len(precision_scores[k]) if precision_scores[k] else 0.0
    
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    
    # Average retrieval time
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
    
    # Total time = Avg Embed Time (Amortized from batches) + Avg Retrieval Time
    avg_total_time = avg_embedding_time + avg_retrieval_time
    
    return {
        'model': model_config['name'],
        'recall_scores': avg_recall,
        'precision_scores': avg_precision,
        'mrr': avg_mrr,
        'avg_embedding_ms': avg_embedding_time, 
        'avg_retrieval_ms': avg_retrieval_time,
        'avg_total_ms': avg_total_time,
        'num_queries': len(queries)
    }


def save_results(results: List[Dict], output_path: str):
    """Save evaluation results to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Get all K values from first result
        k_values = list(results[0]['recall_scores'].keys()) if results else []
        
        fieldnames = (
            ['model', 'num_queries', 'mrr', 'avg_embedding_ms', 'avg_retrieval_ms', 'avg_total_ms'] + 
            [f'recall@{k}' for k in k_values] +
            [f'precision@{k}' for k in k_values]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            row = {
                'model': result['model'],
                'num_queries': result['num_queries'],
                'mrr': f"{result['mrr']:.4f}",
                'avg_embedding_ms': f"{result['avg_embedding_ms']:.2f}",
                'avg_retrieval_ms': f"{result['avg_retrieval_ms']:.2f}",
                'avg_total_ms': f"{result['avg_total_ms']:.2f}"
            }
            for k in k_values:
                row[f'recall@{k}'] = f"{result['recall_scores'][k]:.4f}"
                row[f'precision@{k}'] = f"{result['precision_scores'][k]:.4f}"
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
        print(f"  MRR: {result['mrr']:.4f}")
        for k, score in result['recall_scores'].items():
            print(f"  Recall@{k}: {score:.4f}")
        for k, score in result['precision_scores'].items():
            print(f"  Precision@{k}: {score:.4f}")
        print(f"  Avg Embedding Time: {result['avg_embedding_ms']:.2f} ms")
        print(f"  Avg Retrieval Time: {result['avg_retrieval_ms']:.2f} ms")
        print(f"  Avg Total Time: {result['avg_total_ms']:.2f} ms")


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
    k_values = [1, 5, 10, 15]
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
