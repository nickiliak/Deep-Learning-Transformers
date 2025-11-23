# Exploring Tokenization Strategies for Small-Scale Language Models

## Prerequisites

Before you begin, ensure you have the following installed:

- **PostgreSQL** (v18 or latest): [Download here](https://www.postgresql.org/download/)
- **pgAdmin 4**: [Download here](https://www.pgadmin.org)
- **uv**: A fast Python package installer and resolver.

## Database Setup

1. Open **pgAdmin 4** (or your preferred SQL client).
2. Create a new database with the following details:
   - **Name:** `vectordb`
   - **Port:** `5432`
3. Open the **Query Tool** for the `vectordb` database and run the following command to enable vector support:

```sql
CREATE EXTENSION vector;
```

## Installation & Usage

This project uses uv for dependency management.

1. Install uv

If you haven't installed uv yet, run the appropriate command for your OS:

**macOS**

```
brew install uv
```

**Windows**

```
winget install --id=astral-sh.uv -e
```
2. Setup Project

Sync the environment to install all dependencies:

```
uv sync
```

3. Run the Application

```
uv run main.py
```

or use jupyter notebook, the uv env its going to appear there too

## Overview
Tokenization is a crucial preprocessing step in language modeling, determining how raw text is transformed into units a model can process. This project studies how different tokenization strategies affect the efficiency, cost, and performance of small-scale language models, especially in a Retrieval-Augmented Generation (RAG) setting. We compare tokenizers such as SentencePiece and byte-level tokenization to understand their impact on retrieval accuracy, generation quality, token efficiency, and overall model behavior.

## Motivation
Tokenization affects model performance both directly and indirectly. A tokenizer that produces too many tokens increases computational cost, slows down training and inference, and raises the overall resource requirements. Because small-scale models are more sensitive to token count and efficiency, selecting the right tokenization strategy can significantly improve speed, cost, and linguistic quality. This project aims to uncover these trade-offs.

## Background
This project evaluates how multiple tokenization techniques impact the end-to-end performance of a RAG system using a small transformer-based model. The study includes:
- Training a transformer model on question-answering data
- Implementing retrieval and generation components
- Comparing SentencePiece, byte-level tokenizers, and potentially others
- Measuring their effects on:
  - Retrieval accuracy
  - Generation quality
  - Model efficiency (tokens per input, inference time)
  - Computational cost

Beyond tokenization, the project also explores practical aspects of building and evaluating RAG systems for smaller-scale models.

## Project Milestones
1. Dataset Selection: Identify the most appropriate dataset(s) to support the project goals.
2. Model Training: Train a transformer model using Wikipedia QA data.
3. RAG Integration: Integrate the trained transformer into a Retrieval-Augmented Generation workflow.
4. Evaluation: Assess tokenization efficiency, computational cost, and generation quality across tokenizers.
5. Reporting & Visualization: Compile findings with visualizations, comparisons, and key conclusions.

## Datasets Explored
- SQuAD (Stanford Question Answering Dataset): https://rajpurkar.github.io/SQuAD-explorer/
- Natural Questions (Google Research): https://github.com/google-research-datasets/natural-questions
- TyDi QA: https://github.com/google-research-datasets/tydiqa

## Expected Outcomes
By the end of the project, we aim to deliver:
- A comparison of tokenization techniques for small LMs
- Quantitative metrics on efficiency, generation quality, and retrieval accuracy
- Insights into optimal tokenizer selection for constrained models
- A functioning RAG pipeline with multiple tokenizer configurations
- Clear visualizations and a final written report documenting findings
