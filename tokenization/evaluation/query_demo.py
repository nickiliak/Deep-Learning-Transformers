"""
Simple interactive query tool to test your embeddings.

Usage:
    python query_demo.py "your search query here"
"""

import os
import sys
from dotenv import load_dotenv
from sqlmodel import Session, create_engine, text

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

from tokenization.our_tokenizers.ByT5.ByT5_embedding import ByT5Embedder
from tokenization.our_tokenizers.Canine.Canine_embedding import CanineEmbedder


def search(query: str, model_name: str = "ByT5", k: int = 5):
    """
    Search for documents similar to the query.
    
    Args:
        query: Search query text
        model_name: Which model to use ("ByT5" or "Canine")
        k: Number of results to return
    """
    # Load environment
    load_dotenv()
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://nick:secret@localhost:5433/vectordb"
    )
    
    # Setup model
    if model_name == "ByT5":
        embedder = ByT5Embedder("google/byt5-small")
        table_name = "byt5_small"
    elif model_name == "Canine":
        embedder = CanineEmbedder("google/canine-s")
        table_name = "canine_s"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"🔍 Searching with {model_name}...")
    print(f"Query: '{query}'")
    print("-" * 60)
    
    # Generate query embedding
    query_embedding = embedder.generate_embedding(query)
    
    # Search database
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        sql = text(f"""
            SELECT id, title, text, 
                   1 - (embedding <=> :query_embedding) as similarity
            FROM {table_name}
            ORDER BY embedding <=> :query_embedding
            LIMIT :k
        """)
        
        results = session.exec(sql, {"query_embedding": query_embedding, "k": k})
        
        print(f"\nTop {k} Results:\n")
        for i, (doc_id, title, text, similarity) in enumerate(results, 1):
            print(f"{i}. [{doc_id}] (similarity: {similarity:.4f})")
            print(f"   Title: {title}")
            print(f"   Text: {text[:200]}...")
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_demo.py 'your search query'")
        print("\nExample queries:")
        print("  python query_demo.py 'COVID-19 vaccine effectiveness'")
        print("  python query_demo.py 'machine learning transformers'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    search(query, model_name="ByT5", k=5)
