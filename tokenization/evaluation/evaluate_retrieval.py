"""
Evaluation script for comparing tokenizer/model performance on semantic search.

This script:
1. Connects to the database with embedded documents
2. Takes test queries
3. Retrieves top-k most similar documents using cosine similarity
4. Calculates retrieval metrics (Recall@k, MRR, etc.)
5. Compares different models (BERT, ByT5, Canine)
"""

import os
import sys
from typing import List, Dict, Tuple
import numpy as np
from sqlmodel import Session, create_engine, select, text
from dotenv import load_dotenv

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

from tokenization.our_tokenizers.ByT5.ByT5_embedding import ByT5Embedder
from tokenization.our_tokenizers.Canine.Canine_embedding import CanineEmbedder


class RetrievalEvaluator:
    """Evaluates retrieval performance for different embedding models."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.embedders = {}
        
    def load_embedder(self, model_name: str, model_id: str, embedder_class):
        """Load an embedder for generating query embeddings."""
        print(f"Loading {model_name} embedder...")
        self.embedders[model_name] = embedder_class(model_id)
        
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def retrieve_top_k(self, 
                      table_name: str, 
                      query_embedding: List[float], 
                      k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k most similar documents using pgvector cosine distance.
        
        Returns: List of (doc_id, title, similarity_score)
        """
        with Session(self.engine) as session:
            # pgvector's <=> operator computes cosine distance (1 - cosine_similarity)
            # We order by distance (ascending) to get most similar first
            query = text(f"""
                SELECT id, title, text, 
                       1 - (embedding <=> :query_embedding) as similarity
                FROM {table_name}
                ORDER BY embedding <=> :query_embedding
                LIMIT :k
            """)
            
            result = session.exec(
                query, 
                {"query_embedding": query_embedding, "k": k}
            )
            
            return [(row[0], row[1], float(row[3])) for row in result]
    
    def evaluate_queries(self, 
                        model_name: str,
                        table_name: str,
                        test_queries: List[Dict[str, any]],
                        k: int = 10) -> Dict[str, float]:
        """
        Evaluate retrieval performance on a set of test queries.
        
        test_queries format: [
            {"query": "What is COVID-19?", "relevant_docs": ["doc123", "doc456"]},
            ...
        ]
        
        Returns metrics: recall@k, MRR, precision@k
        """
        embedder = self.embedders[model_name]
        
        recalls = []
        precisions = []
        reciprocal_ranks = []
        
        print(f"\nEvaluating {model_name} on {len(test_queries)} queries...")
        
        for query_item in test_queries:
            query_text = query_item["query"]
            relevant_docs = set(query_item["relevant_docs"])
            
            # Generate query embedding
            query_embedding = embedder.generate_embedding(query_text)
            
            # Retrieve top-k documents
            results = self.retrieve_top_k(table_name, query_embedding, k)
            retrieved_ids = [doc_id for doc_id, _, _ in results]
            
            # Calculate metrics
            relevant_retrieved = set(retrieved_ids) & relevant_docs
            
            # Recall@k: fraction of relevant docs retrieved
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            recalls.append(recall)
            
            # Precision@k: fraction of retrieved docs that are relevant
            precision = len(relevant_retrieved) / k
            precisions.append(precision)
            
            # MRR: reciprocal rank of first relevant document
            rank = next((i + 1 for i, doc_id in enumerate(retrieved_ids) 
                        if doc_id in relevant_docs), 0)
            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0)
        
        return {
            "recall@k": np.mean(recalls),
            "precision@k": np.mean(precisions),
            "MRR": np.mean(reciprocal_ranks),
            "num_queries": len(test_queries)
        }


def main():
    """Example evaluation pipeline."""
    
    # Load environment
    load_dotenv()
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://nick:secret@localhost:5433/vectordb"
    )
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(DATABASE_URL)
    
    # Load embedders
    evaluator.load_embedder("ByT5", "google/byt5-small", ByT5Embedder)
    evaluator.load_embedder("Canine", "google/canine-s", CanineEmbedder)
    # Add BERT if you have it: evaluator.load_embedder("BERT", "bert-base-uncased", BertEmbedder)
    
    # Define test queries (you'd load these from a file in practice)
    test_queries = [
        {
            "query": "What are the symptoms of COVID-19?",
            "relevant_docs": ["doc0", "doc5", "doc12"]  # Example relevant doc IDs
        },
        {
            "query": "How effective are vaccines?",
            "relevant_docs": ["doc3", "doc8"]
        },
        {
            "query": "Natural language processing techniques",
            "relevant_docs": ["doc1", "doc7", "doc15"]
        },
        # Add more queries...
    ]
    
    # Evaluate each model
    results = {}
    
    models_to_test = [
        ("ByT5", "byt5_small"),
        ("Canine", "canine_s"),
        # ("BERT", "bert_all_minilm_l6_v2"),
    ]
    
    for model_name, table_name in models_to_test:
        try:
            metrics = evaluator.evaluate_queries(
                model_name, 
                table_name, 
                test_queries,
                k=10
            )
            results[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Print comparison
    print("\n" + "="*60)
    print("RETRIEVAL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Recall@10':<12} {'Precision@10':<15} {'MRR':<10}")
    print("-"*60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} "
              f"{metrics['recall@k']:<12.4f} "
              f"{metrics['precision@k']:<15.4f} "
              f"{metrics['MRR']:<10.4f}")
    
    print("="*60)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['recall@k'])
    print(f"\n🏆 Best Model (by Recall@10): {best_model[0]}")
    print(f"   Recall: {best_model[1]['recall@k']:.4f}")


if __name__ == "__main__":
    main()
