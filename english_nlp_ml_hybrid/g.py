import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
        self.emotion_mapping = {
            '0': "sad",
            '1': "happy",
            '2': "love",
            '3': "anger",
            '4': "fear",
            '5': "surprise"
        }

    def preprocess_text(self, text):
        """Preprocess the input text by cleaning and basic normalization"""
        # Convert to lowercase
        text = text.lower()

        # Replace contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)

        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)

        return text

    def train_model(self, csv_path='new_dataset.csv'):
        """Train a model based on the input CSV"""
        df = pd.read_csv(csv_path)
        df['emotion'] = df['emotion'].astype(str)  # Ensure 'emotion' column is string

        # Apply preprocessing to sentences
        df['processed_sentence'] = df['sentence'].apply(self.preprocess_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_sentence'], df['emotion'], test_size=0.2, random_state=42
        )

        # Fit the vectorizer and transform text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train a logistic regression model
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X_train_vec, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        print("Model Training Complete!")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.emotion_mapping.values()))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    def add_tfidf(self):
        """Add TF-IDF vectorization to the model"""
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    def add_ngrams(self, min_n=1, max_n=3):
        """Add n-grams (unigrams, bigrams, and trigrams) to the model"""
        self.vectorizer = CountVectorizer(ngram_range=(min_n, max_n))

    def predict_emotion(self, sentence):
        """Predict emotion for a given sentence"""
        processed_sentence = self.preprocess_text(sentence)
        sentence_vec = self.vectorizer.transform([processed_sentence])

        emotion_code = self.model.predict(sentence_vec)[0]
        predicted_emotion = self.emotion_mapping[emotion_code]

        return predicted_emotion

def main():
    detector = EmotionDetector()
  

    # Add TF-IDF vectorization
    detector.add_tfidf()
   

    # Add n-grams (unigrams, bigrams, and trigrams)
    detector.add_ngrams(min_n=1, max_n=3)
    detector.train_model('new_dataset.csv')

    print("Enhanced Emotion Detection System")
    print("Enter 'quit' to exit")

    while True:
        text = input("\nEnter a sentence to analyze: ").strip()

        if text.lower() == 'quit':
            break

        if not text:
            print("Please enter some text to analyze.")
            continue

        emotion = detector.predict_emotion(text)
        print(f"\nPredicted Emotion: {emotion.upper()}")

if __name__ == "__main__":
    main()