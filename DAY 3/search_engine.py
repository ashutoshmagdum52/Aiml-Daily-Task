import os
import re
import string

# ── Stop words ──────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the", "is", "at", "a", "an", "and", "or", "but", "in", "on",
    "of", "to", "for", "with", "by", "from", "it", "its", "this",
    "that", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "not", "as", "their", "they", "we", "he",
    "she", "it", "i", "you", "my", "our", "can", "will", "would",
    "could", "should", "one", "also", "into", "more", "most", "than",
    "so", "if", "about", "which", "when", "who", "how", "what"
}

# ── Synonym map (stretch task) ───────────────────────────────────────────────
SYNONYMS = {
    "happy":      ["joyful", "glad", "pleased", "cheerful"],
    "sad":        ["unhappy", "sorrowful", "depressed", "upset"],
    "fast":       ["quick", "rapid", "swift", "speedy"],
    "big":        ["large", "huge", "massive", "enormous"],
    "smart":      ["intelligent", "clever", "bright"],
    "ai":         ["artificial intelligence", "machine learning", "deep learning"],
    "llm":        ["large language model", "language model", "gpt", "bert"],
    "learn":      ["train", "fine-tune", "study"],
    "data":       ["dataset", "corpus", "information"],
    "code":       ["programming", "script", "software", "development"],
}


def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, and filter stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return tokens


def expand_query(tokens):
    """Expand query tokens with synonyms (Stretch Task)."""
    expanded = set(tokens)
    for token in tokens:
        if token in SYNONYMS:
            for synonym in SYNONYMS[token]:
                # synonyms can be multi-word, so split them too
                for word in synonym.split():
                    expanded.add(word)
    return expanded


def jaccard_similarity(set_a, set_b):
    """Jaccard = intersection / union of two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def load_corpus(filepath):
    """Read corpus.txt and return a list of sentences."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def search(query, corpus_path="corpus.txt", top_n=5):
    """Main search function — returns top N ranked results."""
    sentences = load_corpus(corpus_path)

    # Preprocess the query
    query_tokens = preprocess_text(query)
    query_expanded = expand_query(query_tokens)

    print(f"\nQuery       : {query}")
    print(f"Tokens      : {query_tokens}")
    print(f"After Expand: {sorted(query_expanded)}")
    print("-" * 65)

    results = []
    for sentence in sentences:
        sentence_tokens = set(preprocess_text(sentence))
        score = jaccard_similarity(query_expanded, sentence_tokens)
        results.append((score, sentence))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop {top_n} results:\n")
    for rank, (score, sentence) in enumerate(results[:top_n], start=1):
        print(f"  {rank}. [Score: {score:.4f}] {sentence}")
    print()


if __name__ == "__main__":
    # Build the path relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_file = os.path.join(base_dir, "corpus.txt")

    if not os.path.exists(corpus_file):
        print(f"Error: corpus.txt not found at {corpus_file}")
    else:
        print("=" * 65)
        print("   Mini Search Engine — Keyword & Jaccard Similarity Ranking")
        print("=" * 65)

        # Test queries — shows signal vs noise filtering
        test_queries = [
            "LLM training deep learning",
            "Happy joyful emotion",
            "API security GitHub",
            "data cleaning preprocessing",
        ]

        for query in test_queries:
            search(query, corpus_path=corpus_file, top_n=3)
            print("=" * 65)
