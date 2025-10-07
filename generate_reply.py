import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./model')
tokenizer = GPT2Tokenizer.from_pretrained('./model')
tokenizer.pad_token = tokenizer.eos_token_id

def generate_reply(user_b_message, history=[], max_length=50):
    context = ' '.join(history + [user_b_message])
    input_ids = tokenizer.encode(context, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reply

if __name__ == "__main__":
    user_message = input("User B: ")
    print("User A Reply:", generate_reply(user_message))
