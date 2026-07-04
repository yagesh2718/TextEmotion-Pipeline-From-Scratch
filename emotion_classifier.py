import pandas as pd
import numpy as np
import string
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'emotion'])
    print("Null count:\n", df.isnull().sum())
    
    unique_emotions = df['emotion'].unique()
    print("Unique emotions found:", unique_emotions)
    emotion_nos = {emo: i for i, emo in enumerate(unique_emotions)}
    df['emotion'] = df['emotion'].map(emotion_nos)
    
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower() 
        text = re.sub(r'https?://\S+|www\.\S+', '', text) 
        text = BeautifulSoup(text, "html.parser").get_text() 
        text = text.translate(str.maketrans('', '', string.punctuation)) 
        text = ' '.join([word for word in text.split() if not word.isdigit()]) 
        text = ''.join([char for char in text if char.isascii()]) 
        
        words = word_tokenize(text)
        cleaned_words = [word for word in words if word not in stop_words]
        return ' '.join(cleaned_words)
    
    print("Cleaning text data... This might take a moment.")
    df['text'] = df['text'].apply(clean_text)
    return df

def train_and_evaluate(df):
    X = df['text']
    y = df['emotion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bag of Words Pipeline
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    nb_model_bow = MultinomialNB()
    nb_model_bow.fit(X_train_bow, y_train)
    pred_nb_bow = nb_model_bow.predict(X_test_bow)
    print("Naive Bayes Accuracy (Bag of Words):", accuracy_score(y_test, pred_nb_bow))

    # TF-IDF Pipeline
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    nb_model_tfidf = MultinomialNB()
    nb_model_tfidf.fit(X_train_tfidf, y_train)
    pred_nb_tfidf = nb_model_tfidf.predict(X_test_tfidf)
    print("Naive Bayes Accuracy (TF-IDF):", accuracy_score(y_test, pred_nb_tfidf))

    # Logistic Regression Pipeline
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_tfidf, y_train)
    pred_log = log_model.predict(X_test_tfidf)
    print("Logistic Regression Accuracy (TF-IDF):", accuracy_score(y_test, pred_log))

if __name__ == "__main__":
    dataset_path = 'data/train.txt' 
    try:
        processed_df = load_and_preprocess_data(dataset_path)
        train_and_evaluate(processed_df)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}. Please place 'train.txt' in the 'data' directory.")
