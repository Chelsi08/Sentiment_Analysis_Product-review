import pandas as pd # For data handling
import numpy as np # For numerical operations
import re # For regular expressions (cleaning text)
import nltk # For natural language processing
from nltk.corpus import stopwords # To remove common words like 'the', 'is'

#MachineLearning libraries for the model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text to numbers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Load Data
df = pd.read_csv("archive/1429_1.csv", low_memory=False)
print("Sample data:")
print(df.head())

print("Available columns:")
print(df.columns)

df = df[['reviews.text', 'reviews.rating']].copy()
df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)
df['reviews.text'] = df['reviews.text'].astype(str)
df = df[df['reviews.rating'].isin([1, 3, 5])]
df = df.rename(columns={'reviews.text': 'Text', 'reviews.rating': 'Score'})

def label_sentiment(score):
    if score == 1:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'
df['Sentiment'] = df['Score'].apply(label_sentiment)

nltk.data.path.append("/Users/chelsipatell/Documents/Sentiment_Analysis/stopwords")  
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['Cleaned_Text'] = df['Text'].apply(clean_text)

print("Cleaned and labeled data:")
print(df[['Text', 'Score', 'Sentiment', 'Cleaned_Text']].head())


# Create the TF-IDF vectorizer (to convert text into numeric value)
tfidf = TfidfVectorizer(max_features=3000)  # Keep top 3000 words
# Fit and transform the cleaned text
X = tfidf.fit_transform(df['Cleaned_Text'])
# Encode Sentiment labels (positive, neutral, negative) into numbers
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])
# Show shape of features and labels
print("TF-IDF vector shape:", X.shape)
print("Labels shape:", y.shape)

#Creating Model (Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)

#Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()