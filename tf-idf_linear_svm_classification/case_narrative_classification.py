# TF-IDF Linear SVM Classification
# using CAHOOTS Case Narratives
#
# Script by Aussie Frost
# Updated on Sept 24, 2024

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import export_graphviz
from graphviz import Source
from matplotlib import pyplot as plt

# Define the data path
data_path = '../data/hand_labeled_modes_of_intervention/2023_CAHOOTS_Call_Data_True_Labels.csv'

# Read the data into DataFrame
data = pd.read_csv(data_path)

# Fill missing values
data = data.replace(np.nan, "Other")

# Merge on axis
data['Merged'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

#data.drop('ModeOfIntervention', axis=1)

# Split data into X, y
X, y = data['Merged'], data['ModeOfIntervention']

# Split arrays into random train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

# Define a TF-IDF text vectorization model
tfidf = TfidfVectorizer(max_features = 300, stop_words='english')

# Use TF-IDF to convert X text data into numerical features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Create a LinearSVC classifier
clf = LinearSVC(C=1.0, max_iter=1000)

# Train the model on the training dataset
clf.fit(X_train_tfidf, y_train)

# Predict on the test dataset
y_pred = clf.predict(X_test_tfidf)

# Determine accuracy metrics
accuracy = accuracy_score(y_test, y_pred)

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

# Derive misses
#misses = results[results['y_true'] != results['y_pred']]

# Define path to log file
log_path = "output/dummy_case_narratives_performance.log"

# Log performance results to a .log file
with open(log_path, 'w') as log_file:

    # Log the distribution of true labels
    log_file.write(f"Distribution: \n{data['ModeOfIntervention'].value_counts()}\n\n")

    # Log accuracy score
    log_file.write(f"Model accuracy: {accuracy:.4f}\n\n")
    
    # Log the misses
    #log_file.write("Missed predictions (y_true != y_pred):\n")
    #for index, row in misses.iterrows():
        #log_file.write(f"Index: {index}, X: {row['X']}, y_true: {row['y_true']}, y_pred: {row['y_pred']}\n")

print()
print(f"Log file saved to {log_path}.")

# Define path to export confusion matrix plot to
cm_path = f"output/linear_svm_conf_mat"

# Create and plot a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_.astype(str))
disp.plot(cmap='viridis')
plt.savefig(cm_path)