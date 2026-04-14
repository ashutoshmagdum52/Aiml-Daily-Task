import os
import re
import math
import time
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "gemini-2.5-flash"
DOCUMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "DAY 8", "nexusvortex_doc.txt")
RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "benchmark_results.json")

# The 5 official "Grill" questions from the task
DRILL_QUESTIONS = [
    "I'm seeing Error XV-505 in my terminal. What is the root cause and how do I fix it?",
    "My system is extremely laggy. Which shards should I be worried about and what is the specific command to reset them?",
    "Can I bypass the Theta-Sync requirement to install this on my old i7 laptop?",
    "How does the OS handle user login now that passwords and 2FA are deprecated?",
    "I have a MAC address that was black-holed by the Wraith Firewall. Can I unblock it using the Galactic-Relay?"
]

# Ground-truth reference answers (from the document)
GROUND_TRUTH = [
    "XV-505 is an environment path error. VORTEX_PATH is mapped to /vortex/bin instead of /pulse/bin. Fix: remap to /pulse/bin and verify with vortex --status.",
    "Shard-07 is responsible for system lag. Run vortex --realign-pulse which resets Shard-07 and Shard-04.",
    "Yes via sudo override --bypass-shards but it disables Aegis security and risks Nebula-Leaks corrupting /pulse partition.",
    "Passwords and 2FA are deprecated. Login is via Biometric-Frequency Identification (BFI) using cardiac resonance frequency at 14.2 KHz.",
    "No. Remote unblocking is impossible. A Level-3 Administrator must perform a physical Hardware Key-Turn on the Vortex-Server."
]


def load_and_chunk(filepath, chunk_size):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(" ".join(sentences[i:i + chunk_size]))
    return chunks


def cosine_similarity(vec_a, vec_b):
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query, chunks, embeddings, embed_model, top_k=3):
    query_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
    scores = [(cosine_similarity(query_vec, emb.tolist()), chunks[i])
              for i, emb in enumerate(embeddings)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer ONLY using the provided context. "
    "If not in context, say: I DON'T KNOW. No guessing, no outside knowledge."
)

def generate_answer(question, retrieved_chunks):
    context = "\n\n---\n\n".join([c for _, c in retrieved_chunks])
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    try:
        llm = genai.GenerativeModel(LLM_MODEL, system_instruction=SYSTEM_PROMPT)
        resp = llm.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        return resp.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def score_answer(answer, ground_truth):
    """
    Simple groundedness scoring 1-5:
    5 = All key facts present
    4 = Most facts present, minor omission
    3 = Partial facts, missing one key point
    2 = Vague, mostly off
    1 = Completely wrong or I DON'T KNOW when answer exists
    """
    answer_lower = answer.lower()
    gt_lower = ground_truth.lower()

    # Extract key terms from ground truth and check presence in answer
    key_terms = [w for w in gt_lower.split() if len(w) > 4]
    matches = sum(1 for t in key_terms if t in answer_lower)
    ratio = matches / len(key_terms) if key_terms else 0

    if ratio >= 0.75:   return 5
    elif ratio >= 0.55: return 4
    elif ratio >= 0.35: return 3
    elif ratio >= 0.15: return 2
    else:               return 1


def run_benchmark(chunk_size, label):
    """Run all 5 drill questions with a given chunk size, return scored results."""
    print(f"\n{'='*65}")
    print(f"  CONFIGURATION: {label} (chunk_size={chunk_size})")
    print(f"{'='*65}")

    embed_model = SentenceTransformer(EMBED_MODEL)
    chunks      = load_and_chunk(DOCUMENT_FILE, chunk_size)
    embeddings  = embed_model.encode(chunks, convert_to_tensor=False)

    print(f"  Chunks generated: {len(chunks)}")

    results = []
    total_score = 0

    for i, (question, truth) in enumerate(zip(DRILL_QUESTIONS, GROUND_TRUTH), 1):
        top_chunks = retrieve(question, chunks, embeddings, embed_model)
        answer     = generate_answer(question, top_chunks)
        score      = score_answer(answer, truth)
        total_score += score

        print(f"\n  Q{i}: {question[:70]}...")
        print(f"  Answer: {answer[:120]}...")
        print(f"  Groundedness Score: {score}/5")

        results.append({
            "question": question,
            "answer":   answer,
            "score":    score
        })
        time.sleep(0.5)

    avg = round(total_score / len(DRILL_QUESTIONS), 2)
    print(f"\n  Average Groundedness: {avg}/5")
    return {"config": label, "chunk_size": chunk_size, "avg_score": avg,
            "total_score": total_score, "results": results}


if __name__ == "__main__":
    print("NexusVortex RAG — Benchmarking & Tuning")
    print("Before vs After: chunk_size=3 (small) vs chunk_size=8 (large)")

    # Run both configurations
    config_a = run_benchmark(chunk_size=3, label="Config A — Small Chunks (3 sentences)")
    config_b = run_benchmark(chunk_size=8, label="Config B — Large Chunks (8 sentences)")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_a":  config_a,
        "config_b":  config_b,
        "winner":    "Config A" if config_a["avg_score"] >= config_b["avg_score"] else "Config B"
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=4)

    # Print comparison table
    print(f"\n\n{'='*65}")
    print("  BENCHMARK COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'Question':<6} | {'Config A (chunk=3)':>18} | {'Config B (chunk=8)':>18}")
    print(f"  {'-'*6}-+-{'-'*18}-+-{'-'*18}")
    for i, (ra, rb) in enumerate(zip(config_a["results"], config_b["results"]), 1):
        print(f"  Q{i:<5} | {'Score: '+str(ra['score'])+'/5':>18} | {'Score: '+str(rb['score'])+'/5':>18}")
    print(f"  {'-'*6}-+-{'-'*18}-+-{'-'*18}")
    print(f"  {'AVG':<6} | {str(config_a['avg_score'])+'/5':>18} | {str(config_b['avg_score'])+'/5':>18}")
    print(f"\n  Winner: {output['winner']}")
    print(f"\n  Full results saved to: {RESULTS_FILE}")
