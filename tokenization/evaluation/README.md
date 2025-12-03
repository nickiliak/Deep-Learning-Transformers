# Evaluation Metrics for Retrieval Systems

This document describes the evaluation metrics for comparing tokenization strategies (ByT5, Canine, BPE) in document retrieval tasks.

---

## Currently Implemented ✅

### Recall@K
**Formula:** `Recall@K = (# relevant docs in top-K) / (total # relevant docs)`

**What it measures:** The proportion of relevant documents that appear in the top-K retrieved results.

**Example:**
- Query has 2 relevant documents: `doc0`, `doc1`
- System retrieves top-5: `[doc0, doc5, doc99, doc1, doc42]`
- **Recall@5 = 2/2 = 1.0 (100%)** ✅ Found both relevant docs

**Why it matters:**
- **Most important metric for retrieval** - Did we find what we were looking for?
- Shows system coverage - can it find relevant documents at all?
- Industry standard for search engines and QA systems

**Current K values:** 1, 5, 10

---

## Metrics to Add 📊

### 1. MRR (Mean Reciprocal Rank) ⭐ HIGH PRIORITY

**Formula:** `MRR = average(1 / rank_of_first_relevant_doc)`

**What it measures:** How early the first relevant document appears in results.

**Example:**
- Query 1: First relevant doc at position 1 → RR = 1/1 = 1.0
- Query 2: First relevant doc at position 3 → RR = 1/3 = 0.333
- Query 3: No relevant docs → RR = 0
- **MRR = (1.0 + 0.333 + 0) / 3 = 0.444**

**Why it matters:**
- **User experience metric** - Users typically only look at top 1-3 results
- Complements Recall@K (you can have high recall but low MRR if relevant docs are ranked low)
- Standard metric in QA systems (used by BEIR benchmark)

**Implementation difficulty:** Easy (15-20 minutes)

---

### 2. Precision@K ⭐ MEDIUM PRIORITY

**Formula:** `Precision@K = (# relevant docs in top-K) / K`

**What it measures:** What proportion of retrieved documents are actually relevant?

**Example:**
- Query has 2 relevant documents: `doc0`, `doc1`
- System retrieves top-5: `[doc0, doc5, doc99, doc1, doc42]`
- **Precision@5 = 2/5 = 0.4 (40%)** - Only 40% of retrieved docs are relevant

**Recall vs Precision:**
- **Recall@5 = 1.0** - "Did we find everything?" ✅ Yes
- **Precision@5 = 0.4** - "Is everything we found relevant?" ❌ Only 40%

**Why it matters:**
- Shows retrieval quality, not just quantity
- Important when K is large (e.g., showing 100 results)
- Balances against recall (high recall with low precision = noisy results)

**Implementation difficulty:** Easy (10-15 minutes)

---

### 3. Query Latency (Time per Query) ⭐⭐ HIGH PRIORITY

**What it measures:** Average time to embed a query and retrieve top-K documents.

**Metrics to track:**
- **Embedding time:** Time to convert query text → vector
- **Retrieval time:** Time for pgvector similarity search
- **Total time:** End-to-end query processing

**Why it matters:**
- **Production deployment** - Can this run in real-time?
- ByT5 (1472 dims) vs Canine (768 dims) → Size/speed tradeoff
- Critical for user experience (>100ms feels slow)

**Implementation difficulty:** Very Easy (10 minutes with `time` module)

---

## Recommended Implementation Order

**Day 2 Morning (2-3 hours):**
1. ✅ **MRR** (20 min) - Essential metric, easy to add
2. ✅ **Latency tracking** (15 min) - Easy, great for discussion
3. ✅ **Precision@K** (15 min) - Complements recall nicely
4. ✅ **Update CSV output** (30 min) - Add new columns
5. ✅ **Create visualization script** - Bar charts, comparison tables

---

## Expected Results Analysis

**Current findings:**
- **ByT5:** 0% recall → Model not suited for retrieval (seq2seq architecture)
- **Canine:** 15.5% recall@10 → Works but mediocre
- **BPE:** 0% recall → Untrained LSTM (random weights)

**What additional metrics will reveal:**

**MRR:**
- Canine likely has low MRR (~0.05-0.10) → Relevant docs ranked low
- Explains why recall@1 is 5% but recall@10 is 15.5%

**Latency:**
- ByT5 probably slowest (1472 dims + encoder complexity)
- Canine middle ground (768 dims)
- BPE might be fastest (simple LSTM, 768 dims)