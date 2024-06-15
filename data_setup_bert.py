import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['HF_TOKEN'] = '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'

def load_data(file_path):
    data = pd.read_csv(file_path)
    texts = data['review_text'].astype(str).tolist()
    labels = data['not_recommended'].values
    return texts, labels

def split_data(texts, labels, test_size=0.2, random_state=42):
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)
