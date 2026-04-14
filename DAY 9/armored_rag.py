import os
import re
import math
import time
import textwrap
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "gemini-2.5-flash"
CHUNK_SIZE    = 5
TOP_K         = 3
MAX_QUERY_LEN = 300
DOCUMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "DAY 8", "nexusvortex_doc.txt")
LOG_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "performance.log")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ── Input Validator ────────────────────────────────────────────────────────────

class InputValidator:
    """Intercepts and validates all user queries before processing."""

    def validate(self, query):
        """
        Returns (is_valid: bool, reason: str)
        Rule 1: Reject empty or only-special-character strings
        Rule 2: Reject queries longer than 300 characters
        """
        # Rule 1: Empty or only special characters / whitespace
        stripped = query.strip()
        if not stripped:
            return False, "Query is empty. Please enter a question."

        # Check if only special characters remain after removing alphanumerics/spaces
        only_special = re.sub(r'[a-zA-Z0-9\s]', '', stripped)
        if len(only_special) == len(stripped):
            return False, "Query contains only special characters. Please enter a real question."

        # Rule 2: Token stuffing — query too long
        if len(stripped) > MAX_QUERY_LEN:
            return False, (
                f"Query too long ({len(stripped)} chars). "
                f"Maximum allowed is {MAX_QUERY_LEN} characters."
            )

        return True, "OK"


# ── Chunking ──────────────────────────────────────────────────────────────────

def load_and_chunk(filepath, chunk_size=CHUNK_SIZE):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks, model):
    print(f"  Building vector store: {len(chunks)} chunks...")
    return model.encode(chunks, convert_to_tensor=False)


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query, chunks, embeddings, embed_model):
    query_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_vec, emb.tolist())
        scores.append((score, chunks[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:TOP_K]


# ── Generation with Graceful Failure ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise technical assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, respond with: I DON'T KNOW. "
    "Do not guess. No outside knowledge."
)

def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join([chunk for _, chunk in retrieved_chunks])
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"

    try:
        llm = genai.GenerativeModel(LLM_MODEL, system_instruction=SYSTEM_PROMPT)
        response = llm.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        return response.text.strip()
    except Exception as e:
        # Graceful failure — no Python traceback shown to user
        logging.error(f"API Error: {e}")
        return "Service temporarily unavailable. Please try again in 30 seconds."


# ── Main Armored Pipeline ─────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_FILE):
        print(f"Error: nexusvortex_doc.txt not found.\n  Expected: {DOCUMENT_FILE}")
        exit(1)

    validator = InputValidator()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Chunking document and building vector store...")
    chunks     = load_and_chunk(DOCUMENT_FILE)
    embeddings = build_vector_store(chunks, embed_model)

    print(f"\nArmored RAG Pipeline — Ready")
    print(f"Validation: max {MAX_QUERY_LEN} chars | no empty/special-only queries")
    print(f"Logging to: {LOG_FILE}")
    print("=" * 65)
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        # ── Validation gate ──────────────────────────────────────────────────
        is_valid, reason = validator.validate(question)
        if not is_valid:
            print(f"\n[REJECTED] {reason}\n")
            logging.warning(f"REJECTED query: '{question[:80]}' | Reason: {reason}")
            continue

        # ── Retrieval with latency logging ───────────────────────────────────
        t0 = time.time()
        top_chunks = retrieve(question, chunks, embeddings, embed_model)
        retrieval_ms = round((time.time() - t0) * 1000, 2)

        # ── Generation with latency logging ─────────────────────────────────
        t1 = time.time()
        answer = generate_answer(question, top_chunks)
        generation_ms = round((time.time() - t1) * 1000, 2)

        # ── Log performance ──────────────────────────────────────────────────
        log_entry = (
            f"query='{question[:60]}' | "
            f"retrieval_time_ms={retrieval_ms} | "
            f"generation_time_ms={generation_ms}"
        )
        logging.info(log_entry)

        # ── Display ──────────────────────────────────────────────────────────
        print(f"\n[Retrieved {len(top_chunks)} chunks | "
              f"Retrieval: {retrieval_ms}ms | Generation: {generation_ms}ms]")
        for i, (score, chunk) in enumerate(top_chunks, 1):
            preview = textwrap.shorten(chunk, width=75, placeholder="...")
            print(f"  {i}. [Score: {score:.4f}] {preview}")
        print(f"\nBot: {answer}\n")
