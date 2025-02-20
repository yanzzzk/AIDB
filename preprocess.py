import os
import pandas as pd

DATA_DIR = "/Users/yanly/AIDB/aclImdb/train"

# read
def load_imdb_data(folder):
    reviews = []
    labels = []
    scores = []
    for sentiment in ["pos", "neg"]:
        folder_path = os.path.join(DATA_DIR, sentiment)
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                score = int(filename.split("_")[1].split(".")[0])
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                    review = file.read().strip()
                reviews.append(review)
                labels.append(sentiment) 
                scores.append(score)
    df = pd.DataFrame({"review": reviews, "label": labels, "score": scores})
    return df

# do data
df = load_imdb_data("train")
# save
df.to_csv("reviews.csv", index=False)

print("complete, save to reviews")
