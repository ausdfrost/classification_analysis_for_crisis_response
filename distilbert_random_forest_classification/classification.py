# DistilBERT Random Forest Classification
# Script by Aussie Frost
# Updated on Sept 18, 2024

import numpy as np
import pandas as pd

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Read the data into DataFrame
data = pd.read_csv("../data/dummy_case_narratives.csv")

# Split data into X, y
X, y = data['x'], data['y']

# Split arrays into random train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

# Use TF-IDF to convert X text data into numerical features
tfidf = TfidfVectorizer(max_features = 100)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define text encoder
def text_encoder(texts, tokenizer, model):
    encoded_inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # Use mean pooling to get sentence embeddings
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Tokenize the text data and convert to tensors
X_train_distilbert = text_encoder(X_train, tokenizer, model)
X_test_distilbert = text_encoder(X_test, tokenizer, model)

# Create a RF classifier
clf = RandomForestClassifier(n_estimators = 100)

# Train the model on the training dataset
clf.fit(X_train_tfidf, y_train)

# Predict on the test dataset
y_pred = clf.predict(X_test_tfidf)

# Determine accuracy metrics
accuracy = metrics.accuracy_score(y_test, y_pred)

# Create new DF with test results
results = pd.DataFrame(
    {
        'X': X_test,
        'y_true': y_test,
        'y_pred': y_pred
    }
)

# Output the test results to a CSV
results.to_csv('output/dummy_case_narratives_results.csv')

# Output misses
misses = results[results['y_true'] != results['y_pred']]

# Define path to log file
log_path = "output/dummy_case_narratives_performance.log"

# Log both accuracy and misses to a .log file
with open(log_path, 'w') as log_file:

    # Log accuracy score
    log_file.write(f"Model accuracy: {accuracy:.2f}\n\n")
    
    # Log the misses
    log_file.write("Missed predictions (y_true != y_pred):\n")
    for index, row in misses.iterrows():
        log_file.write(f"Index: {index}, X: {row['X']}, y_true: {row['y_true']}, y_pred: {row['y_pred']}\n")

print()
print(f"Log file saved to {log_path}.")