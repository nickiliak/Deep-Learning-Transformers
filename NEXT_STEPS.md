# TODO

## Critical (Must Complete)

### Model
1. **Fix Canine embedder** - Has circular import bug in `Canine_embedding.py` line 4
2. **Implement BERT embedder** - Use `sentence-transformers/all-MiniLM-L6-v2` (384 dims, subword tokenization)
   - Skip SentencePiece (redundant with BERT/WordPiece, both are subword methods)
   - This gives 3 distinct strategies: Byte (ByT5) vs Character (Canine) vs Subword (BERT)

### Pipeline
1. **Run pipeline for all 3 models** - Currently only ByT5 tested, need Canine + BERT embeddings
2. Auto-loop through all embedders (optional, saves manual config changes)
3. GPU batching optimization (only if generation is too slow)

## High Priority

### Tokenization Evaluation
1. **Add evaluation metrics:**
   - Token length & compression efficiency
   - OOV (out-of-vocabulary) rate
   - Robustness to typos/rare words
2. **Dimensionality differences (384 vs 768 vs 1472):**
   - **Option A:** Accept different dims, normalize with accuracy-per-dimension metric
   - **Option B:** Apply PCA to reduce all to same dimension (e.g., 256)
   - **Recommendation:** Start with A (simpler), only use PCA if results are clearly skewed by capacity

### Data Selection
1. **NQ dataset decision:** Keep it (standard benchmark, 3.6M docs, 3,452 queries)

## Nice to Have

### RAG + Inference
1. Connect `groq_inference.py` to retrieval pipeline (retrieve → inject context → generate)
2. **Note:** Lower priority - project focuses on retrieval, not generation

---

## Execution Order
1. Fix Canine bug (5 min)
2. Implement BERT embedder (30 min)
3. Run pipeline 3x for all models (2-3 hours)
4. Run evaluation scripts (30 min)
5. Enhanced analysis & write-up (3-4 hours)

**Success = 3 working embedders + quantitative comparison + analysis**
