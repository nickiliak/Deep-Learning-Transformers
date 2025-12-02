# Exploring Tokenization Strategies for Small-Scale Language Models

## Project Overview

This project compares different tokenization strategies (byte-level, character-level, and subword) in the context of semantic search and retrieval. We use three different transformer models with their respective tokenization approaches:

- **BERT (MiniLM)**: WordPiece tokenization (384 dimensions)
- **ByT5**: Byte-level tokenization (1472 dimensions)
- **Canine**: Character-level tokenization (768 dimensions)

The pipeline generates embeddings for a document corpus using each model, stores them in a PostgreSQL database with pgvector, and provides tools to evaluate retrieval performance across tokenization strategies.

## Repository Structure

```
pipeline/                    # Main embedding generation pipeline
  └── pipeline.ipynb        # Notebook to generate and store embeddings
tokenization/
  ├── our_tokenizers/       # Embedder implementations
  │   ├── ByT5/            # Byte-level tokenization
  │   ├── Canine/          # Character-level tokenization
  │   └── BPE/             # Subword tokenization (BERT)
  └── evaluation/          # Tools to evaluate retrieval performance
      ├── evaluate_retrieval.py  # Comprehensive comparison
      └── query_demo.py          # Interactive search demo
setup/
  ├── setup.py             # Downloads Natural Questions dataset
  └── docker-compose.yml   # PostgreSQL + pgvector database
data/                      # Dataset storage (created by setup.py)
```

## Setup Instructions

### 1. Install Prerequisites

**uv (Python Package Manager)**

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

- **macOS**: `brew install uv`
- **Windows**: `winget install --id=astral-sh.uv -e`
- **Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Docker Desktop**

Required to run the PostgreSQL database locally.

- Download and install: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Start Docker Desktop after installation

### 2. Install Python Dependencies

From the repository root, sync the environment:

```bash
uv sync
```

This creates a virtual environment and installs all required packages (PyTorch, Transformers, pgvector, etc.).

**Important: Install PyTorch with CUDA support:**

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

This installs PyTorch 2.6+ with CUDA 12.1 support (required for GPU acceleration and security fixes).

### 3. Download Dataset

Run the setup script to download the Natural Questions dataset from BEIR:

```bash
cd setup
uv run setup.py
```

This downloads and extracts the NQ dataset (~500MB) into the `data/` folder. The dataset contains queries and a document corpus for evaluation.

### 4. Start Database

Spin up PostgreSQL with pgvector extension:

```bash
cd setup
docker-compose up -d
```

This creates:
- PostgreSQL database on `localhost:5433`
- Enables pgvector extension for similarity search
- Creates a `.env` file at the repository root with connection string

To stop the database: `docker-compose down`

To view logs: `docker-compose logs -f`

## Usage

### Generate Embeddings

Open `pipeline/pipeline.ipynb` in VS Code or Jupyter. The notebook allows you to:

1. Select which embedder to use (ByT5, Canine, or BERT)
2. Configure the model ID and vector dimensions
3. Run the pipeline to generate embeddings for all documents
4. Store embeddings in PostgreSQL with pgvector indexing

Run the pipeline separately for each model to populate the database with all three embedding sets.

### Evaluate Retrieval Performance

After generating embeddings, use the evaluation tools:

**Interactive Search** (test a single query):
```bash
cd tokenization/evaluation
uv run query_demo.py "your search query here"
```

**Comprehensive Evaluation** (compare all models):
```bash
cd tokenization/evaluation
uv run evaluate_retrieval.py
```

This compares all three tokenization strategies using metrics like Recall@k, MRR (Mean Reciprocal Rank), and query latency.

## Project Background

Tokenization is a crucial preprocessing step that determines how text is split into units a model can process. This project investigates how different tokenization approaches affect retrieval performance:

- **Byte-level (ByT5)**: No vocabulary limit, handles any text including rare characters and misspellings
- **Character-level (Canine)**: Works at character granularity, robust to typos and morphological variations
- **Subword (BERT)**: Balances vocabulary size with linguistic meaning using WordPiece tokenization

### Research Questions

1. Does more granular tokenization (bytes/characters) improve retrieval for out-of-vocabulary terms?
2. How do embedding dimensions affect retrieval quality across tokenization strategies?
3. What are the trade-offs between tokenization granularity, computational cost, and accuracy?

### Dataset

We use the **Natural Questions** dataset from BEIR:
- 3.6M+ Wikipedia passages
- 3,452 real Google search queries
- Ground truth relevance judgments for evaluation

Source: [BEIR Benchmark](https://github.com/beir-cellar/beir)
