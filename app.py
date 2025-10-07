from flask import Flask, request, jsonify

import joblib

from transformers import GPT2Tokenizer


app = Flask(__name__)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

model = joblib.load("Model.joblib")


@app.route("/chat", methods=["POST"])

def chat():

    user_message = request.json.get("message", "")

    input_text = f"User B: {user_message}\nUser A:"

    inputs = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(inputs, max_length=60, pad_token_id=tokenizer.eos_token_id,

                            do_sample=True, top_k=50, top_p=0.95)

    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"reply": reply})


if __name__ == "__main__":

    app.run(debug=True)