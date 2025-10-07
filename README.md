# Offline Chat-Reply Recommendation System

## Description
Predicts User A's reply in two-person conversations using GPT-2 transformer.

## Folder Structure
- `dataset/` → Contains user chat CSV files.
- `model/` → Fine-tuned GPT-2 model will be saved here.
- `ChatRec_Model.ipynb` → Jupyter notebook with preprocessing, training, evaluation, and reply generation.
- `utils.py` → Helper functions for preprocessing.
- `generate_reply.py` → Script to generate replies offline.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run `ChatRec_Model.ipynb` to preprocess and train GPT-2 offline.
3. Run `generate_reply.py` to generate replies using trained model.
