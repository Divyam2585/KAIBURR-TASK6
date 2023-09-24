# KAIBURR-TASK6
Performing a text classification task on the Consumer Complaint Database involves several steps. Below is a high-level overview of the steps you should follow and a summary of the code for each step. Please note that this is a simplified example, and in practice, you might need to fine-tune your approach and consider more advanced techniques.

**Step 1: Explanatory Data Analysis and Feature Engineering**

In this step, you'll explore the dataset, understand its structure, and perform feature engineering if needed. You'll also prepare the data for text processing.

```python
# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv("consumer_complaints.csv")

# Explore the dataset
print(df.head())

# Check for missing values and handle them if necessary
df = df.dropna(subset=["Consumer complaint narrative"])

# Feature Engineering: You can create additional features if needed
# For example, you can extract text length, word count, etc.
df["text_length"] = df["Consumer complaint narrative"].apply(len)

# Create target labels
category_mapping = {
    "Credit reporting, repair, or other": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}
df["Category"] = df["Product"].map(category_mapping)
```

**Step 2: Text Pre-Processing**

Text data needs to be pre-processed before using it for model training. Common steps include lowercasing, tokenization, and removing stopwords and special characters.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["Consumer complaint narrative"],
    df["Category"],
    test_size=0.2,
    random_state=42
)

# Create a TF-IDF vectorizer for text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

**Step 3: Selection of Multi Classification Model**

You can choose a multi-class classification model, such as Multinomial Naive Bayes or a deep learning model like a neural network. Here, we'll use a simple Multinomial Naive Bayes classifier.

```python
from sklearn.naive_bayes import MultinomialNB

# Create and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
```

**Step 4: Comparison of Model Performance**

You can compare the performance of multiple models using various metrics like accuracy, precision, recall, and F1-score. For simplicity, we'll use accuracy here.

```python
# Evaluate the classifier on the test data
accuracy = classifier.score(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy}")
```

**Step 5: Model Evaluation**

You can evaluate the model using confusion matrices, classification reports, and other metrics to get a more detailed view of its performance.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the test data
y_pred = classifier.predict(X_test_tfidf)

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print(class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```

**Step 6: Prediction and Summary**

Finally, you can use the trained model to make predictions on new consumer complaints.

```python
# Example prediction
new_complaint = ["I have an issue with my mortgage payment."]
new_complaint_tfidf = tfidf_vectorizer.transform(new_complaint)
predicted_category = classifier.predict(new_complaint_tfidf)[0]

# Map the predicted category back to its label
predicted_label = [k for k, v in category_mapping.items() if v == predicted_category][0]

print(f"Predicted Category: {predicted_label}")
```

This is a simplified example of text classification. In practice, you may need to experiment with different models, hyperparameters, and perform more in-depth analysis and validation to build a robust consumer complaint classifier.
