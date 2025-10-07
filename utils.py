import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z0-9?.!,]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
