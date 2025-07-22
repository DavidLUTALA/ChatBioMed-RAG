# scripts/evaluation.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Utilise PubMedBERT pour cohérence
embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def evaluate_similarity(answer: str, sources: list[str]) -> float:
    """Calcule la similarité cosinus entre la réponse et les sources concaténées."""
    source_text = " ".join(sources)
    
    embeddings = embedder.encode([answer, source_text])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return round(sim_score, 4)
