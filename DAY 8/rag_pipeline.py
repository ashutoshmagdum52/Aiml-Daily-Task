import os
import math
import textwrap
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "gemini-2.5-flash"
CHUNK_SIZE    = 5       # sentences per chunk
TOP_K         = 3       # top chunks to retrieve
DOCUMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexusvortex_doc.txt")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── Step 1: Chunking ──────────────────────────────────────────────────────────

def load_and_chunk(filepath, chunk_size=CHUNK_SIZE):
    """Split document into overlapping sentence chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into sentences on period/newline
    import re
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ── Step 2: Vector Store (in-memory) ──────────────────────────────────────────

def build_vector_store(chunks, model):
    """Encode all chunks into embeddings — our in-memory vector store."""
    print(f"  Building vector store: {len(chunks)} chunks...")
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings   # list of numpy arrays


# ── Step 3: Retrieval — Cosine Similarity ─────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query, chunks, embeddings, model, top_k=TOP_K):
    """Encode query, find top-k most similar chunks."""
    query_vec = model.encode(query, convert_to_tensor=False).tolist()
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_vec, emb.tolist())
        scores.append((score, chunks[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# ── Step 4: Generation — Grounded LLM Answer ─────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise technical assistant. "
    "Answer the user's question using ONLY the provided context below. "
    "Do not use any outside knowledge. "
    "If the answer is not present in the context, respond with exactly: I DON'T KNOW. "
    "Do not guess. Do not add information not found in the context."
)

def generate_answer(question, retrieved_chunks):
    """Build context from top chunks and send to LLM."""
    context = "\n\n---\n\n".join([chunk for _, chunk in retrieved_chunks])

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    model = genai.GenerativeModel(LLM_MODEL, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        user_message,
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )
    return response.text.strip()


# ── Main RAG Loop ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_FILE):
        print(f"Error: Document not found at:\n  {DOCUMENT_FILE}")
        exit(1)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Chunking document...")
    chunks = load_and_chunk(DOCUMENT_FILE)

    print("Building vector store...")
    embeddings = build_vector_store(chunks, embed_model)

    print(f"\nRAG system ready. Document: nexusvortex_doc.txt")
    print(f"Chunks: {len(chunks)}  |  Embedding model: {EMBED_MODEL}")
    print(f"LLM: {LLM_MODEL}  |  Top-K retrieval: {TOP_K}")
    print("=" * 65)
    print("Ask questions about the NexusVortex OS document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Exiting RAG pipeline.")
            break

        # Retrieval
        top_chunks = retrieve(question, chunks, embeddings, embed_model)

        print(f"\n[Retrieved {len(top_chunks)} chunks]")
        for i, (score, chunk) in enumerate(top_chunks, 1):
            preview = textwrap.shorten(chunk, width=80, placeholder="...")
            print(f"  {i}. [Score: {score:.4f}] {preview}")

        # Generation
        print("\nBot:", end=" ", flush=True)
        answer = generate_answer(question, top_chunks)
        print(answer)
        print()
