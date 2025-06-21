from flask import Flask, request, jsonify
from flask_cors import CORS
from immigrationAssistant import build_agent, ask_question

app = Flask(__name__)
CORS(app)

agent = build_agent()

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    topic = data.get("topic")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = ask_question(agent, question, topic)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
