import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# Max exchanges to remember (sliding window)
MAX_MEMORY = 5

# System persona — never breaks character
SYSTEM_PROMPT = (
    "You are Alex, a sharp and professional Technical Recruiter at a top AI company. "
    "You only discuss topics related to tech recruitment, job roles, skills, resumes, "
    "interview preparation, and career advice in the technology industry. "
    "If the user asks anything unrelated to tech careers or recruitment, politely redirect "
    "them back to the topic. Never break character under any circumstances. "
    "Keep responses concise and professional."
)

model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)


def get_response(conversation_history):
    """Send full conversation history to the LLM and get a reply."""
    response = model.generate_content(
        conversation_history,
        generation_config=genai.types.GenerationConfig(temperature=0.7)
    )
    return response.text.strip()


def apply_sliding_window(history, max_exchanges=MAX_MEMORY):
    """Keep only the last N user+assistant exchange pairs."""
    # Each exchange = 1 user message + 1 assistant message = 2 items
    max_messages = max_exchanges * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


def run_chatbot():
    conversation_history = []

    print("=" * 60)
    print("  Technical Recruiter Bot — Alex (Powered by Groq/LLaMA3)")
    print("  Context window: last 5 exchanges only")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    print("\nAlex: Hi! I'm Alex, your Technical Recruiter. How can I help")
    print("      you today? Are you looking for a job, preparing for an")
    print("      interview, or need advice on your tech career?\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nAlex: Good luck with your career! Feel free to come back anytime.")
            break

        # Add user message to history
        conversation_history.append({"role": "user", "parts": [user_input]})

        # Apply sliding window BEFORE sending (trim old context)
        conversation_history = apply_sliding_window(conversation_history)

        try:
            reply = get_response(conversation_history)
        except Exception as e:
            reply = f"[Error contacting API: {e}]"

        # Add assistant reply to history
        conversation_history.append({"role": "model", "parts": [reply]})

        print(f"\nAlex: {reply}\n")

        # Show memory status for transparency
        exchanges = len(conversation_history) // 2
        print(f"[Memory: {exchanges}/{MAX_MEMORY} exchanges stored]\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file.")
    else:
        run_chatbot()
