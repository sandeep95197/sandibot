# chatbot_filtered.py
import sys
from ctransformers import AutoModelForCausalLM

# Fix Unicode issue in Windows console
sys.stdout.reconfigure(encoding='utf-8')

# ---------------- Configuration ----------------
MODEL_PATH = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
BOT_NAME = "SandiBot"
MAX_TOKENS = 50
FILTER_WORDS = ["violence", "drugs", "hate"]  # Add forbidden words

# ---------------- Load Model ----------------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama"
)
print("Model loaded!")
print(f"{BOT_NAME} is ready! Type 'bye' to exit.\n")

# ---------------- Chat Loop ----------------
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["bye", "exit"]:
        print(f"{BOT_NAME}: Goodbye! Have a great day.")
        break

    # Build prompt
    prompt = (
        f"You are {BOT_NAME}, a friendly AI assistant. "
        f"Answer briefly, naturally, and like a human.\n"
        f"User: {user_input}\n"
        f"{BOT_NAME}:"
    )

    # Generate response
    response = ""
    for token in model(
        prompt,
        max_new_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        stream=True
    ):
        response += token
        # Stop generation if too long
        if len(response.split()) > 30:
            break

    # Strip whitespace and check for filtered words
    response = response.strip()
    if any(word in response.lower() for word in FILTER_WORDS):
        response = "Sorry, I cannot answer that."

    print(response)
