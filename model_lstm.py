import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

df = pd.read_csv("/Users/yanly/AIDB/reviews.csv")
df['label'] = df['label'].map({'pos': 1, 'neg': 0})

# 2. train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 3. Tokenization + GloVe 
tokenizer = get_tokenizer("basic_english")
vocab = GloVe(name="6B", dim=100)
word2idx = dict(vocab.stoi)
if "<unk>" not in word2idx:
    word2idx["<unk>"] = 0

MAX_LEN = 200  
BATCH_SIZE = 16

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, word2idx, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        token_ids = [self.word2idx.get(token, self.word2idx.get("<unk>", 0)) for token in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids), torch.tensor(self.labels[idx])

train_dataset = IMDbDataset(train_texts, train_labels, word2idx, tokenizer)
test_dataset = IMDbDataset(test_texts, test_labels, word2idx, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 5. LSTM
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 1  
NUM_LAYERS = 1 
EPOCHS = 3  
LR = 0.0005 

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return self.sigmoid(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train_model():
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            texts, labels = texts.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

train_model()

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predictions = (outputs > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

evaluate_model()

def predict_sentiment(text):
    model.eval()
    tokens = tokenizer(text)
    token_ids = [word2idx.get(token, word2idx.get("<unk>", 0)) for token in tokens]
    if len(token_ids) < MAX_LEN:
        token_ids += [0] * (MAX_LEN - len(token_ids))
    else:
        token_ids = token_ids[:MAX_LEN]
    input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        sentiment = "Positive" if output.item() > 0.5 else "Negative"
        return sentiment

sample_review = "This movie was absolutely fantastic! The storyline was great and the acting was top-notch."
print(f"Sample Review Sentiment: {predict_sentiment(sample_review)}")

# 6.save model
torch.save(model.state_dict(), "model_lstm.pth")
with open("vectorizer_lstm.pkl", "wb") as f:
    pickle.dump(word2idx, f)

print("✅ LSTM Save！")
