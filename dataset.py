from datasets import load_dataset
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
dataset = load_dataset("batterydata/pos_tagging")

# Convert train and test sets to Pandas DataFrames
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Use only the first 1000 rows for training
train_df = train_df.iloc[:1000]

# Debugging: check columns and structure
print("Column names in train_df:", train_df.columns)
print(train_df.head())

# Update column names
word_column = "words"  # Change from "word" to "words"
pos_column = "labels"  # Change from "pos" to "labels"

# Extract words and POS tags
sentences = train_df[word_column].tolist()
pos_tags = train_df[pos_column].tolist()

# Flatten the pos_tags list to get one tag for each word, aligned with words in the training set
flat_pos_tags = [tag for sublist in pos_tags for tag in sublist]  # Flatten the list
flat_words = [word for sublist in sentences for word in sublist]  # Flatten the list of words

# Check the flattening process
print(f"Flattened POS tags length: {len(flat_pos_tags)}")
print(f"Flattened words length: {len(flat_words)}")
print(f"First 10 POS tags: {flat_pos_tags[:10]}")
print(f"First 10 words: {flat_words[:10]}")

# Convert words to Word2Vec Embeddings
# Train Word2Vec model on the sentences (list of words per sentence)
print("Training Word2Vec model...")
word2vec = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Function to get word embedding (return a zero vector if word is not in vocabulary)
def get_embedding(word):
    if word in word2vec.wv:
        return word2vec.wv[word]
    else:
        return np.zeros(100)  # Return a zero vector for out-of-vocabulary words

# Convert each word in the sentences to its corresponding word embedding
train_X = np.array([get_embedding(word) for word in flat_words])

# Encode POS tags
print("Encoding POS tags...")
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(flat_pos_tags)  # Train labels

# For the test set
test_sentences = test_df[word_column].tolist()
test_flat_words = [word for sentence in test_sentences for word in sentence]
test_flat_pos_tags = [tag for sublist in test_df["test"]["labels"] for tag in sublist]

# Convert each word in the test sentences to its corresponding word embedding
test_X = np.array([get_embedding(word) for word in test_flat_words])

# Encode POS tags for the test set
test_y = label_encoder.transform(test_flat_pos_tags)

# Train Logistic Regression or SVM
print("Training classifier...")
clf = LogisticRegression(max_iter=1000)  # or SVC()
clf.fit(train_X, train_y)

# Predict on test set
test_pred = clf.predict(test_X)

# Evaluate the performance
accuracy = accuracy_score(test_y, test_pred)
print(f"Accuracy: {accuracy:.4f}")

