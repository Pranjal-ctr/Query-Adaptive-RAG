"""
Interactive Terminal Interface for Query-Adaptive RAG
Allows switching between Yelp and Legal domains and testing the adaptive K-selection in real-time.

Usage:
    python src/cli.py --domain yelp
    python src/cli.py --domain legal
    python src/cli.py  # (Defaults to yelp)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import spacy
import numpy as np

from config import (
    PROJECT_ROOT, EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS,
    EMBEDDING_ENCODE_KWARGS, LLM_MODEL, BGE_QUERY_PREFIX,
)
from intent_classifier import classify_intent
from regime_aware_retrieval import load_classifier, extract_features_for_query, K_FLAT, K_DISCRIMINATIVE
from rag_query import format_docs

# Paths
DB_YELP = os.path.join(PROJECT_ROOT, "faiss_food_db")
DB_LEGAL = os.path.join(PROJECT_ROOT, "data", "legal", "faiss_index")

PROMPT = PromptTemplate(
    template="""You are an expert analytical assistant.
Answer the user's question strictly using the provided context. If the answer is not in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)


def load_vectorstore(domain: str):
    """Load FAISS vectorstore for specific domain."""
    db_path = DB_YELP if domain == "yelp" else DB_LEGAL
    if not os.path.exists(db_path):
        print(f"\n  [!] Error: FAISS database not found at {db_path}")
        sys.exit(1)
        
    print(f"  Loading {domain.upper()} vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


def retrieve_with_scores(vectorstore, query: str, k: int):
    prefixed_query = BGE_QUERY_PREFIX + query
    results = vectorstore.similarity_search_with_score(prefixed_query, k=k)
    
    converted = []
    for doc, score in results:
        similarity = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        converted.append((doc, similarity))
    converted.sort(key=lambda x: x[1], reverse=True)
    return converted


def main():
    parser = argparse.ArgumentParser(description="Interactive Query-Adaptive RAG Terminal")
    parser.add_argument("--domain", type=str, choices=["yelp", "legal"], default="yelp",
                        help="Choose the dataset to query (yelp or legal)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"  Query-Adaptive RAG Interactive CLI — Domain: {args.domain.upper()}")
    print("=" * 60)

    # Load components
    vectorstore = load_vectorstore(args.domain)
    
    print("  Loading LLM (Ollama Mistral)...")
    llm = Ollama(model=LLM_MODEL)
    
    print("  Loading Regime Classifier (spaCy & Sklearn)...")
    classifier_data = load_classifier()
    model = classifier_data["model"]
    feature_keys = classifier_data["feature_keys"]
    nlp = spacy.load("en_core_web_sm")
    
    print("\n  [ READY ] Type your query below. Type 'exit' to quit.\n")

    while True:
        try:
            query = input(f"[{args.domain.upper()}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            break

        # 1. Feature Extraction & Classifier Prediction
        feats = extract_features_for_query(query, nlp)
        X = np.array([[feats[k] for k in feature_keys]])
        predicted_regime = model.predict(X)[0] 
        
        regime_label = "Flat" if predicted_regime == 0 else "Discriminative"
        k_selected = K_FLAT if predicted_regime == 0 else K_DISCRIMINATIVE

        # 2. Retrieval
        print(f"\n  [Classifier] Predicted Regime: {regime_label} -> Adjusting mapping to K={k_selected}...")
        results = retrieve_with_scores(vectorstore, query, k=k_selected)
        
        if not results:
            print("  [!] No context retrieved.")
            continue
            
        # 3. Generate Answer
        print(f"  [Retrieval] Found {len(results)} chunks. Generating answer...\n")
        context = format_docs(results, k=k_selected)
        response = llm.invoke(PROMPT.format(context=context, question=query))
        
        # Output
        print("-" * 60)
        print("🤖 RAG Answer:")
        print(response)
        
        # Stats Output
        print("-" * 60)
        print("📊 Retrieval Stats:")
        print(f"  • Top Chunk Similarity: {results[0][1]:.4f}")
        print(f"  • Lowest Chunk Similarity: {results[-1][1]:.4f}")
        print(f"  • Score Range: {results[0][1] - results[-1][1]:.4f}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
