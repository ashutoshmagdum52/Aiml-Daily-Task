import os
import math

# sentence-transformers uses the all-MiniLM-L6-v2 model locally
from sentence_transformers import SentenceTransformer, util

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "DAY 3", "corpus.txt")
TOP_N = 3


def load_corpus(filepath):
    """Load sentences from corpus.txt."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def cosine_similarity(vec_a, vec_b):
    """Manual cosine similarity: dot(a,b) / (|a| * |b|)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def search(query, corpus_sentences, corpus_embeddings, model, top_n=TOP_N):
    """Encode query, compute cosine similarity against all corpus embeddings."""
    query_embedding = model.encode(query, convert_to_tensor=False).tolist()

    scores = []
    for i, sentence_embedding in enumerate(corpus_embeddings):
        score = cosine_similarity(query_embedding, sentence_embedding.tolist())
        scores.append((score, corpus_sentences[i]))

    # Sort descending by similarity score
    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"\nQuery: \"{query}\"")
    print("-" * 65)
    for rank, (score, sentence) in enumerate(scores[:top_n], 1):
        print(f"  {rank}. [Score: {score:.4f}] {sentence}")
    print()


if __name__ == "__main__":
    # Check corpus file exists
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: corpus.txt not found at:\n  {CORPUS_FILE}")
        print("Make sure DAY 3/corpus.txt exists.")
        exit(1)

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading corpus...")
    corpus = load_corpus(CORPUS_FILE)
    print(f"  {len(corpus)} sentences loaded.")

    print("Generating embeddings for all sentences (one-time step)...")
    # Returns numpy arrays — we store as list for cosine_similarity
    corpus_embeddings = model.encode(corpus, convert_to_tensor=False)
    print("  Embeddings ready.\n")

    print("=" * 65)
    print("  Semantic Search Engine — Sentence Embeddings + Cosine Similarity")
    print("=" * 65)

    # Test queries — semantic search finds related concepts, not just exact words
    test_queries = [
        "Processor chip for running neural networks",   # should find GPU/CPU sentences
        "How models learn from examples",               # fine-tuning / training
        "Keeping secrets safe in code",                 # API key security
        "Understanding words in a sentence",            # attention / transformers
        "Feeling happy and joyful",                     # sentiment / emotion (noise)
    ]

    for query in test_queries:
        search(query, corpus, corpus_embeddings, model, top_n=TOP_N)
        print("=" * 65)

    # Interactive mode
    print("\nEnter your own queries (type 'quit' to exit):\n")
    while True:
        user_query = input("Search: ").strip()
        if user_query.lower() in ("quit", "exit", ""):
            break
        search(user_query, corpus, corpus_embeddings, model, top_n=TOP_N)
