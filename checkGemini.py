import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize client (it automatically looks for GEMINI_API_KEY in .env)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
#sample request
#{
#  "message": "Hello Gemini! Can you explain how a Mac uses Unix?"
#}

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    try:
        # Using the ultra-fast flash model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)