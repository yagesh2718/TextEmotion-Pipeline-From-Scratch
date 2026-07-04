# Text Emotion Classification Pipeline

This repository contains a Natural Language Processing (NLP) project that classifies text into various emotions using Machine Learning algorithms. It serves as a comprehensive guide to text preprocessing, feature extraction, and traditional ML models in NLP.

## 🚀 Features

* **Complete Text Preprocessing Pipeline:**
  * Lowercasing
  * URL and HTML Tag removal (using `BeautifulSoup`)
  * Punctuation, Number, and Emoji removal
  * Stopword removal (using `NLTK`)
* **Feature Extraction Techniques:**
  * Bag of Words (BoW)
  * Term Frequency-Inverse Document Frequency (TF-IDF)
* **Machine Learning Models:**
  * Multinomial Naive Bayes
  * Logistic Regression
* **NLP Core Concepts Demo:**
  * Separate script demonstrating Stemming, Lemmatization, and One-Hot Encoding.

## 📂 Repository Structure

* `emotion_classifier.py` - The main script containing data cleaning, preprocessing, and model training/evaluation.
* `nlp_basics_demo.py` - An educational script showcasing raw implementations of BoW, TF-IDF, Stemming, and Lemmatization.
* `data/train.txt` - Text file containing the dataset (semicolon separated).
* `requirements.txt` - List of dependencies required to run the project.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Emotion-Classification-NLP.git](https://github.com/your-username/Emotion-Classification-NLP.git)
   cd Emotion-Classification-NLP
