"""
Google NQ Dataset Preprocessing
Filters corpus to 3000 documents and aligns queries with filtered documents.
"""

import pandas as pd
import os
from pathlib import Path


def preprocess_data(corpus_size=3000, output_dir="../data_filtered"):
    """
    Preprocess the Natural Questions dataset.
    
    Args:
        corpus_size: Number of unique documents to keep (default: 3000)
        output_dir: Directory to save filtered data (default: ../data_filtered)
    
    Returns:
        tuple: (corpus_file_path, queries_file_path)
    """
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    print("Loading corpus dataset...")
    corpus_path = script_dir / '../data/nq/corpus.jsonl'
    df_nqcorpus = pd.read_json(corpus_path, lines=True)
    print(f"  Total documents: {len(df_nqcorpus)}")
    print(f"  Unique titles: {df_nqcorpus['title'].nunique()}")
    
    # Filter corpus to first N documents
    print(f"\nFiltering corpus to {corpus_size} documents...")
    df_nqcorpus_filtered = df_nqcorpus[
        df_nqcorpus["title"].isin(df_nqcorpus["title"].unique()[:corpus_size])
    ]
    df_nqcorpus_filtered = df_nqcorpus_filtered.drop(columns=["metadata"])
    print(f"  Filtered documents: {len(df_nqcorpus_filtered)}")
    print(f"  Unique titles: {df_nqcorpus_filtered['title'].nunique()}")
    
    # Save filtered corpus
    corpus_output_file = output_path / 'corpus_filtered.jsonl'
    df_nqcorpus_filtered.to_json(
        corpus_output_file,
        lines=True,
        orient='records'
    )
    print(f"  Saved to: {corpus_output_file}")
    
    # Load queries
    print("\nLoading queries dataset...")
    queries_path = script_dir / '../data/nq/queries.jsonl'
    df_nqqueries = pd.read_json(queries_path, lines=True)
    print(f"  Total queries: {len(df_nqqueries)}")
    
    # Load relevance judgments
    print("Loading relevance judgments...")
    qrels_path = script_dir / '../data/nq/qrels/test.tsv'
    df_nqtest = pd.read_csv(qrels_path, sep='\t')
    print(f"  Total query-corpus pairs: {len(df_nqtest)}")
    
    # Merge queries with relevance judgments
    print("\nMerging queries with relevance judgments...")
    df_nqqueries_merge = df_nqqueries.merge(
        df_nqtest.groupby('query-id')['corpus-id'].apply(list).reset_index(),
        left_on='_id',
        right_on='query-id',
        how='left'
    )
    df_nqqueries_merge = df_nqqueries_merge.drop(columns=['metadata', 'query-id'])
    
    # Filter to keep only queries where ALL corpus-ids exist in filtered corpus
    print("Filtering queries to match filtered corpus...")
    df_nqqueries_merge_filtered = df_nqqueries_merge[
        df_nqqueries_merge['corpus-id'].apply(
            lambda ids: all(id_ in df_nqcorpus_filtered['_id'].values for id_ in ids)
        )
    ]
    print(f"  Filtered queries: {len(df_nqqueries_merge_filtered)}")
    
    # Save filtered queries
    queries_output_file = output_path / 'queries_filtered.jsonl'
    df_nqqueries_merge_filtered.to_json(
        queries_output_file,
        lines=True,
        orient='records'
    )
    print(f"  Saved to: {queries_output_file}")
    
    return str(corpus_output_file), str(queries_output_file)


if __name__ == "__main__":
    print("="*80)
    print("PREPROCESSING: Google NQ Dataset")
    print("="*80 + "\n")
    
    try:
        corpus_file, queries_file = preprocess_data()
        print("\n" + "="*80)
        print("[OK] Preprocessing completed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


