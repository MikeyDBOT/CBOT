from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the conversational pipeline with a small model (CPU-friendly)
chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    responses = chatbot(user_message)
    reply = responses[0]["generated_text"]
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
