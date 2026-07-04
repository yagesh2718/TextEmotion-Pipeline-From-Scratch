import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

print("--- 1. ONE HOT ENCODING ---")
vocabulary = ["akarsh", "watch", "sheryians", "harsh", "also", "teach"]
word_to_ind = {word: i for i, word in enumerate(vocabulary)}
print("Word indices:", word_to_ind)

sentence = "harsh also watch sheryians"
words = sentence.split()
ohm = np.zeros((len(words), len(vocabulary)), dtype=int)

for i, word in enumerate(words):
    if word in word_to_ind:
        idx = word_to_ind[word]
        ohm[i, idx] = 1
print("One-Hot Matrix:\n", ohm)


print("\n--- 2. BAG OF WORDS (BoW) ---")
corpus_bow = [
    "I love pizza. Pizza is the best.",
    "I love pasta.",
    "Pasta is great."
]
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_bow)
print("Vocabulary:", bow_vectorizer.get_feature_names_out())
print("BoW Array:\n", X_bow.toarray())


print("\n--- 3. TF-IDF ---")
corpus_tfidf = [
    "Akarsh watch Sheryians",
    "Harsh also watch Sheryians",
    "Sheryians teach Sheryians"
]
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_tfidf)
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Array:\n", X_tfidf.toarray())


print("\n--- 4. STEMMING ---")
ps = PorterStemmer()
def stem_text(text):
    words = word_tokenize(text)
    return " ".join([ps.stem(word) for word in words])

sample_text_1 = "I am walking to the store while he walked and she walks."
print("Original:", sample_text_1)
print("Stemmed: ", stem_text(sample_text_1))


print("\n--- 5. LEMMATIZATION ---")
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text, pos_tag='v'):
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word, pos=pos_tag) for word in words])

sample_text_2 = "The striped mice are running and eating."
print("Original Text:       ", sample_text_2)
print("Lemmatized (Noun):   ", lemmatize_text(sample_text_2, pos_tag='n'))
print("Lemmatized (Verb):   ", lemmatize_text(sample_text_2, pos_tag='v'))
