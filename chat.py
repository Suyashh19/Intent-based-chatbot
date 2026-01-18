import pickle
import sys

# =========================
# 1. Load Pre-Trained Model
# =========================
try:
    # Ensure these files exist in the same directory
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'model.pkl' or 'vectorizer.pkl' not found.")
    print("   Please ensure the trained model files are in the current directory.")
    sys.exit(1)

print("\n🤖 Intent Recognizer (Runtime Only)")
print("   Ready for commands. Type 'quit' to exit.\n")

# =========================
# 2. Command Recognition Loop
# =========================
while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C gracefully
        print("\nBot: Stopping program.")
        break

    if user_input.lower() == "quit":
        print("Bot: Terminating.")
        break

    if not user_input:
        continue

    # Vectorize input using the frozen vocabulary
    X = vectorizer.transform([user_input])

    # Predict intent
    # The model forces the input into one of the known classes
    predicted_intent = model.predict(X)[0]

    print(f"Command: {predicted_intent}")