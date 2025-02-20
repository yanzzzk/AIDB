import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pickle

# 1. 读取 IMDb 数据
df = pd.read_csv("/Users/yanly/AIDB/reviews.csv")
df['label'] = df['label'].map({'pos': 1, 'neg': 0})

# 2. 划分训练集与测试集（80% / 20%）
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["review"], padding="max_length", truncation=True, max_length=256)

# 4. 将 Pandas DataFrame 转换为 HuggingFace Dataset 对象
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
# 移除不必要的列
train_dataset = train_dataset.remove_columns(["review", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["review", "__index_level_0__"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 5. 加载预训练的 BERT 模型（2 类）
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. 定义训练参数
training_args = TrainingArguments(
    output_dir="./bert_output",
    evaluation_strategy="epoch",  
    num_train_epochs=3,  # 建议微调3个 epoch，确保模型充分训练
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    disable_tqdm=False,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. 开始训练
trainer.train()

# 8. 在测试集上评估模型
eval_results = trainer.evaluate()
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# 9. 示例预测函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

sample_review = "This movie was absolutely fantastic! The storyline was great and the acting was top-notch."
print(f"Sample Review Sentiment: {predict_sentiment(sample_review)}")

# 10. 保存微调后的模型和 Tokenizer
torch.save(model.state_dict(), "model_bert.pth")
with open("vectorizer_bert.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ BERT save!")
