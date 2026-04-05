# Query-Adaptive RAG 

This repository contains the codebase and findings for the **Query-Adaptive Retrieval-Augmented Generation (RAG)** empirical study. 

The project demonstrates that **similarity score distributions produced by vector retrieval are structured, query-dependent signals** whose shape varies systematically across domains. It implements a regime-aware retrieval pipeline that predicts the optimal number of chunks (K) to retrieve based on the query's complexity, overcoming the limitations of traditional fixed-K RAG systems.

## Project Structure

The project encompasses a full end-to-end pipeline across two distinct domains (Yelp Consumer Reviews and Legal Statutes):

- **`src/`**: Contains the core Python scripts for ingestion, querying, and analysis.
  - `ingest.py` & `ingest_legal.py`: Data loaders for populating the FAISS vector database.
  - `generate_queries.py` & `generate_legal_queries.py`: Generates domain-specific Q&A pairs using Mistral 7B.
  - `rag_query.py`: Core RAG execution script.
  - `analyze_curves.py`: Computes statistical features (entropy, kurtosis, etc.) from similarity curves.
  - `compare_domains.py`: Performs cross-domain t-test statistical comparisons.
  - `regime_classifier.py`: Trains an NLP-based pre-retrieval classifier (Logistic Regression) to predict optimal retrieval strategy.
  - `regime_aware_retrieval.py`: Executes the adaptive K-selection retrieval based on classifier predictions.
- **`dashboard/`**: A standalone Vite + Vanilla JS single-page application that statically loads the JSON results to present rich, interactive analytics, charts, and theoretical frameworks.
- **`data/`**: Stores raw documents and query datasets.
- **`results/`**: Outputs from the analytical pipeline including JSON reports, feature data, and generated taxonomy plots.

## Key Findings

1. **Domain Structure Dictates Retrieval Regime:** The linguistic and semantic properties of a corpus fundamentally determine the shape of similarity score distributions. 
2. **Discriminative vs. Flat Regimes:** 
   - **Legal Texts** naturally produce a highly *discriminative* regime (high peak similarity, wide score range, steep decay), meaning fewer K chunks (K=3-5) are required. 
   - **Consumer Reviews** (Yelp) yield a *flat* regime (lower peak similarity, gradual decay), requiring broader context aggregation (K=8-12).
3. **Regime Prediction:** We achieved an accuracy of ~67% in predicting the optimal retrieval regime using a lightweight classifier trained solely on pre-retrieval NLP features extracted from the query using spaCy.

## Setup and Installation

### Prerequisites

- Python 3.9+
- Node.js (for the frontend dashboard)
- Ollama running locally with the `mistral` model (for query generation and RAG answering)

### Backend Setup

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS/Linux
   source myenv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

### Frontend Dashboard Setup

1. Navigate to the dashboard directory:
   ```bash
   cd dashboard
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the dashboard at `http://localhost:5173/`.

## Author
[Pranjal-ctr](https://github.com/Pranjal-ctr)
