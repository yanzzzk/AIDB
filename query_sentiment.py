import sqlite3
import pandas as pd
import pickle
import numpy as np
import torch

# pick model
USE_BERT = False

#  SQLite
conn = sqlite3.connect("imdb.db")

df = pd.read_csv("reviews.csv")
df.to_sql("reviews", conn, if_exists="replace", index=False)

def get_all_reviews():
    cursor = conn.cursor()
    cursor.execute("SELECT review FROM reviews")
    rows = cursor.fetchall()
    return [row[0] for row in rows]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_BERT:
    from transformers import BertTokenizer, BertForSequenceClassification
    with open("vectorizer_bert.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load("model_bert.pth", map_location=device))
    model.to(device)
    model.eval()

    def predict_sentiments(texts, batch_size=32):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1] 
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)
else:
    from model_lstm import LSTMModel
    with open("vectorizer_lstm.pkl", "rb") as f:
        word2idx = pickle.load(f)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 1
    NUM_LAYERS = 1
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("model_lstm.pth", map_location=device))
    model.eval()

    def lstm_predict(texts, max_len=200, batch_size=32):
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            token_ids_list = []
            for text in batch_texts:
                tokens = text.split()
                ids = [word2idx.get(token, word2idx.get("<unk>", 0)) for token in tokens]
                if len(ids) < max_len:
                    ids += [0] * (max_len - len(ids))
                else:
                    ids = ids[:max_len]
                token_ids_list.append(ids)
            inputs = torch.tensor(token_ids_list).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                all_scores.extend(outputs.cpu().numpy().squeeze())
        return np.array(all_scores)
    
    predict_sentiments = lstm_predict

# search
def query_sentiment():
    texts = get_all_reviews()
    if len(texts) == 0:
        print("no data。")
        return
    scores = predict_sentiments(texts, batch_size=32)
    avg_score = np.mean(scores)
    print(f"result, average score is {avg_score:.4f}")
query_sentiment()

def query_sentiment_approx(sample_ratio=0.1, batch_size=32):
    texts = get_all_reviews()
    sample_size = int(len(texts) * sample_ratio)
    if sample_size < 1:
        sample_size = 1
    sampled_texts = np.random.choice(texts, sample_size, replace=False)
    scores = predict_sentiments(list(sampled_texts), batch_size=batch_size)
    avg_score = np.mean(scores)
    std_dev = np.std(scores)
    ci = 1.96 * std_dev / np.sqrt(sample_size)
    print(f"Approximate query results: The average sentiment score is {avg_score:.4f} ± {ci:.4f}")
query_sentiment_approx()
conn.close()
