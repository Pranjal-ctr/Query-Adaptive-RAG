<p align="center">
  <h1 align="center">🔍 Query-Adaptive RAG</h1>
  <p align="center"><em>Cross-Domain Regime-Aware Retrieval-Augmented Generation</em></p>
  <p align="center"><strong>A research project demonstrating that similarity score distributions are structured, predictive signals for optimizing RAG retrieval depth.</strong></p>
</p>

---

## 👥 Team Members

| Name | Role | Responsibilities |
|------|------|------------------|
| **Pranjal Sahani** | Team Leader / Lead Developer | Project architecture, RAG pipeline design, regime-aware retrieval algorithm, experiment orchestration, cross-domain statistical analysis |
| **Ankit Pandey** | Data Engineering & Pipeline | Data ingestion (Yelp & Legal), FAISS indexing, query generation, score curve collection, data preprocessing |
| **Anil Gurjar** | Frontend & Visualization | Analytics dashboard (Vite + Chart.js), interactive visualizations, cross-domain comparison plots, taxonomy charts |
| **Shreyas Sanjay** | Literature Review & Evaluation | Research survey, RAG evaluation framework, regime classifier training & analysis, documentation and report writing |

---

## 📌 Project Overview

Standard RAG systems retrieve a **fixed number of K** chunks regardless of query complexity or domain. This project proves empirically that the **shape of the similarity score distribution** itself is a predictive signal. We:

1. **Characterize** distribution shapes using entropy, kurtosis, score range, and drop ratios across two domains — **Yelp (consumer reviews)** and **Legal (Indian Penal Code statutes)**
2. **Classify** queries into *Flat* vs *Discriminative* regimes using a pre-retrieval NLP classifier (66.9% accuracy, 0.715 AUC)
3. **Adapt** retrieval depth (K) dynamically per query, proving that domain structure governs optimal retrieval strategy

### Key Results
- **4 out of 10 metrics** show statistically significant differences (p < 0.005) between domains
- Legal queries are **65% discriminative** (optimal K=3–5); Yelp queries are **59% flat** (optimal K=8–12)
- Syntactic depth (dependency tree) is the **strongest predictor** of retrieval regime

---

## 📁 Repository Structure

```
Query-Adaptive-RAG/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── src/                             # Core Python source code
│   ├── config.py                    # Centralized configuration (paths, models, params)
│   ├── ingest.py                    # Yelp data ingestion & FAISS indexing
│   ├── ingest_legal.py              # Legal statute ingestion & FAISS indexing
│   ├── generate_queries.py          # Yelp query generation (Mistral LLM)
│   ├── generate_legal_queries.py    # Legal query generation (Mistral LLM)
│   ├── intent_classifier.py         # Rule-based intent classification
│   ├── score_logger.py              # Similarity score curve logger (CSV/JSON)
│   ├── rag_query.py                 # RAG pipeline with score collection
│   ├── analyze_curves.py            # Statistical feature extraction from curves
│   ├── evaluate_rag.py              # RAG evaluation framework (metrics)
│   ├── compare_domains.py           # Cross-domain t-test statistical comparison
│   ├── regime_classifier.py         # ML classifier (LogReg/RF/GBM) training
│   ├── regime_aware_retrieval.py    # Adaptive K-selection evaluation
│   ├── run_legal_analysis.py        # End-to-end legal pipeline orchestrator
│   └── cli.py                       # Interactive terminal for querying either domain
│
├── data/                            # Raw datasets & generated queries
│   ├── yelp_reviews_sample.jsonl    # Yelp consumer review corpus
│   ├── sample_queries_research.json # Generated Yelp queries with intents
│   ├── score_curves/                # Yelp similarity score curves
│   │   └── curve_data.json
│   └── legal/                       # Legal domain data
│       ├── chunks.jsonl             # Indian Penal Code statute chunks
│       ├── faiss_index/             # Legal FAISS vector database
│       ├── queries.json             # Generated legal queries
│       └── score_curves/            # Legal similarity score curves
│
├── faiss_food_db/                   # Yelp FAISS vector database
│
├── models/                          # Trained ML models
│   └── regime_classifier.pkl        # Serialized regime classifier
│
├── results/                         # Experiment outputs
│   ├── curve_features.json          # Extracted statistical features (Yelp)
│   ├── comparison_table.json        # Cross-domain t-test results
│   ├── classifier_results.json      # Classifier accuracy, AUC, confusion matrix
│   ├── regime_aware_evaluation.json # Baseline vs regime-aware comparison
│   ├── cross_domain_comparison.png  # Statistical comparison visualization
│   ├── score_curves_overview.png    # Score distribution overview plot
│   ├── yelp/                        # Yelp-specific results
│   │   └── evaluation_report.json
│   └── legal/                       # Legal-specific results
│       └── evaluation_report.json
│
├── eval/                            # RAG quality evaluation outputs
│   ├── rag_evaluation_results.json
│   ├── rag_evaluation_summary.json
│   └── rag_evaluation_report.txt
│
├── dashboard/                       # Frontend analytics dashboard
│   ├── index.html                   # Entry point
│   ├── package.json                 # Node.js dependencies
│   ├── src/
│   │   ├── main.js                  # Dashboard application logic (Chart.js)
│   │   └── style.css                # Glassmorphism dark-theme styles
│   └── public/
│       ├── data/                    # Symlinked/copied JSON results for UI
│       └── img/                     # Generated analytical plots
│
└── walkthrough_phase0.md            # Phase 0 baseline walkthrough
```

---

## ⚙️ Installation

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.9+ | Backend pipeline |
| **Node.js** | 18+ | Frontend dashboard |
| **Ollama** | Latest | Local LLM inference |
| **Git** | Latest | Version control |

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pranjal-ctr/Query-Adaptive-RAG.git
cd Query-Adaptive-RAG
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python -m venv myenv

# Activate it
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (required for regime classifier)
python -m spacy download en_core_web_sm
```

### Step 3: Ollama Setup (for LLM inference)

```bash
# Install Ollama from https://ollama.ai
# Pull the Mistral model
ollama pull mistral

# Start the Ollama server (runs on localhost:11434)
ollama serve
```

### Step 4: Frontend Dashboard Setup

```bash
cd dashboard
npm install
```

---

## 🚀 Executing the Code

### A. Full Pipeline Execution (Sequential Steps)

Run these commands from the project root directory with the virtual environment activated:

#### Phase 1 — Data Ingestion

```bash
# Ingest Yelp reviews into FAISS
python src/ingest.py

# Ingest Legal statutes into FAISS
python src/ingest_legal.py
```

#### Phase 2 — Query Generation

```bash
# Generate Yelp domain queries (requires Ollama running)
python src/generate_queries.py

# Generate Legal domain queries
python src/generate_legal_queries.py
```

#### Phase 3 — RAG Retrieval & Score Collection

```bash
# Run RAG pipeline on Yelp queries (collects similarity score curves)
python src/rag_query.py --batch data/sample_queries_research_filtered.txt
```

#### Phase 4 — Curve Analysis & Feature Extraction

```bash
# Extract statistical features from score distributions
python src/analyze_curves.py
```

#### Phase 5 — Cross-Domain Evaluation

```bash
# Evaluate RAG quality metrics
python src/evaluate_rag.py

# Run full legal pipeline (ingestion → curves → evaluation)
python src/run_legal_analysis.py

# Statistical comparison between domains (t-tests)
python src/compare_domains.py
```

#### Phase 6 — Regime Classifier Training

```bash
# Train the pre-retrieval regime classifier
python src/regime_classifier.py
```

#### Phase 7 — Regime-Aware Retrieval

```bash
# Evaluate adaptive K-selection vs fixed K baseline
python src/regime_aware_retrieval.py
```

---

### B. Interactive Query Mode

Once the pipeline data exists, you can interactively query either domain:

```bash
# Query the Yelp dataset
python src/cli.py --domain yelp

# Query the Legal dataset
python src/cli.py --domain legal
```

**What happens when you type a query:**
1. The **regime classifier** analyzes your query text using spaCy NLP features
2. It **predicts** whether the query is *Flat* or *Discriminative*
3. It dynamically sets **K** (K=12 for flat, K=4 for discriminative)
4. It **retrieves** chunks from FAISS and passes them to Mistral
5. It prints the **answer** along with retrieval statistics (similarity scores, score range)

**Example session:**
```
[LEGAL] > What is the punishment for robbery under Section 310 of IPC?

  [Classifier] Predicted Regime: Discriminative -> Adjusting mapping to K=4...
  [Retrieval] Found 4 chunks. Generating answer...

------------------------------------------------------------
🤖 RAG Answer:
According to Section 310 of IPC, robbery or dacoity when armed with
a deadly weapon carries a minimum sentence of...
------------------------------------------------------------
📊 Retrieval Stats:
  • Top Chunk Similarity: 0.8240
  • Lowest Chunk Similarity: 0.7113
  • Score Range: 0.1127
============================================================
```

---

### C. Launch the Analytics Dashboard

```bash
cd dashboard
npm run dev
```

Open **http://localhost:5173/** in your browser to explore:

- 📊 **Overview** — KPIs, query distribution, theoretical framework
- 🍽️ **Yelp Domain** — Similarity distributions, feature radar charts
- ⚖️ **Legal Domain** — Taxonomy plots, intent clustering
- 🔄 **Cross-Domain Comparison** — t-test table, bar charts, domain structure analysis
- 🤖 **Classifier** — Model accuracy, confusion matrix, feature importance
- 🎯 **Regime-Aware Retrieval** — Baseline vs adaptive K performance

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Embeddings** | BAAI/bge-base-en-v1.5 (sentence-transformers) |
| **Vector Database** | FAISS (faiss-cpu, IndexFlatIP) |
| **LLM** | Mistral 7B (via Ollama, localhost) |
| **NLP Features** | spaCy (en_core_web_sm) |
| **ML Classifiers** | scikit-learn (LogisticRegression, RandomForest, GradientBoosting) |
| **Visualization** | matplotlib, seaborn (backend) · Chart.js (frontend) |
| **Frontend** | Vite + Vanilla JavaScript |
| **Styling** | Custom CSS (glassmorphism dark theme) |

---

## 📄 License

This project is developed for academic and research purposes.

---

<p align="center">
  <sub>Built with ❤️ by the Query-Adaptive RAG Research Team</sub>
</p>
