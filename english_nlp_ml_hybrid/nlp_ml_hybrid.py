import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class EmotionDetector:
    def __init__(self):
        self.initialize_dictionaries()
        self.model = None
        self.vectorizer = CountVectorizer()
        self.emotion_mapping = {
            '0': "sad",
            '1': "happy",
            '2': "love",
            '3': "anger",
            '4': "fear",
            '5': "surprise"
        }

    def initialize_dictionaries(self):
        """Initialize emotion dictionaries with comprehensive word lists"""
        self.emotion_words = {
            'happy': {
                # Core happiness words
                'happy': 1.0, 'joy': 1.0, 'delighted': 0.9, 'excited': 0.9, 'wonderful': 0.9,
                'great': 0.8, 'pleased': 0.8, 'glad': 0.8, 'fantastic': 0.9, 'thrilled': 0.9,
                'enjoy': 0.8, 'cheerful': 0.8, 'blessed': 0.8, 'smile': 0.7, 'laugh': 0.8,
                'content': 0.7, 'satisfied': 0.7, 'peaceful': 0.6, 'proud': 0.7, 'fun': 0.7,
                'awesome': 0.9, 'amazing': 0.9, 'excellent': 0.8, 'loving': 0.8, 'thankful': 0.7,
                'grateful': 0.8, 'bliss': 0.9, 'joy': 0.9, 'celebrate': 0.8, 'win': 0.7,
                'success': 0.7, 'perfect': 0.8, 'beautiful': 0.7, 'bright': 0.6
            },
            'sad': {
                # Core sadness words
                'sad': 1.0, 'unhappy': 0.9, 'depressed': 0.9, 'miserable': 0.9, 'heartbroken': 1.0,
                'disappointed': 0.8, 'upset': 0.8, 'hurt': 0.8, 'crying': 0.9, 'cry': 0.8,
                'tears': 0.8, 'grief': 0.9, 'grieving': 0.9, 'sorrow': 0.9, 'pain': 0.8,
                'painful': 0.8, 'alone': 0.7, 'lonely': 0.8, 'lost': 0.7, 'broken': 0.8,
                'sorry': 0.6, 'regret': 0.7, 'hopeless': 0.9, 'despair': 0.9, 'fail': 0.7,
                'failed': 0.7, 'failure': 0.7, 'miss': 0.6, 'missing': 0.6, 'empty': 0.8
            },
            'anger': {
                # Core anger words
                'angry': 1.0, 'mad': 0.9, 'furious': 1.0, 'rage': 1.0, 'hate': 0.9,
                'frustrate': 0.8, 'frustrated': 0.8, 'irritated': 0.7, 'annoyed': 0.7, 'upset': 0.7,
                'bitter': 0.8, 'disgusted': 0.8, 'disgusting': 0.8, 'hostile': 0.9, 'anger': 1.0,
                'outraged': 0.9, 'offensive': 0.7, 'aggressive': 0.8, 'fury': 1.0, 'wrathful': 0.9,
                'threatening': 0.8, 'violent': 0.9, 'destroy': 0.8, 'revenge': 0.8, 'cruel': 0.8
            },
            'fear': {
                # Core fear words
                'afraid': 1.0, 'scared': 1.0, 'frightened': 1.0, 'terrified': 1.0, 'panic': 0.9,
                'fear': 1.0, 'worried': 0.8, 'worry': 0.8, 'anxious': 0.8, 'nervous': 0.8,
                'horrified': 0.9, 'horror': 0.9, 'terrible': 0.8, 'terror': 1.0, 'scary': 0.8,
                'scared': 1.0, 'frightening': 0.9, 'afraid': 1.0, 'dread': 0.9, 'stressed': 0.7,
                'stress': 0.7, 'paranoid': 0.8, 'suspicious': 0.6, 'doubt': 0.6, 'uneasy': 0.7
            },
            'surprise': {
                # Core surprise words
                'surprised': 1.0, 'wow': 0.9, 'amazing': 0.8, 'astonished': 1.0, 'shocked': 0.9,
                'unexpected': 0.8, 'sudden': 0.7, 'startled': 0.8, 'surprise': 1.0, 'stunning': 0.8,
                'incredible': 0.8, 'unbelievable': 0.9, 'wonder': 0.7, 'wonderful': 0.7, 'magical': 0.7,
                'miraculous': 0.8, 'spectacular': 0.8, 'amazed': 0.9, 'extraordinary': 0.7, 'odd': 0.6
            },
            'love': {
                # Core love words
                'love': 1.0, 'adore': 0.9, 'cherish': 0.9, 'passionate': 0.8, 'romantic': 0.8,
                'affectionate': 0.8, 'caring': 0.7, 'lovely': 0.7, 'sweetest': 0.8, 'darling': 0.8,
                'devoted': 0.9, 'beloved': 0.9, 'precious': 0.8, 'intimate': 0.8, 'close': 0.7,
                'loved': 1.0, 'loves': 1.0, 'loving': 0.9, 'romance': 0.8, 'beautiful': 0.7,
                'tender': 0.7, 'gentle': 0.6, 'warm': 0.6, 'together': 0.6, 'forever': 0.7
            }
        }

        # Initialize intensifiers with their multipliers
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'so': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
            'completely': 1.8, 'totally': 1.8, 'utterly': 1.8, 'deeply': 1.8, 'incredibly': 2.0,
            'intensely': 1.8, 'particularly': 1.3, 'quite': 1.3, 'rather': 1.2, 'somewhat': 0.8
        }

        # Initialize negators
        self.negators = {'not', 'never', 'no', "n't", 'cannot', 'cant', 'wont', 'wouldnt', 'shouldnt', 'couldnt'}

        # Add word variations to emotion dictionaries
        self._add_word_variations()

    def _add_word_variations(self):
        """Add common variations of emotional words"""
        for emotion, words in self.emotion_words.items():
            variations = {}
            for word, score in words.items():
                # Add common verb forms
                variations[word + 'ing'] = score
                variations[word + 'ed'] = score
                variations[word + 's'] = score
                # Add common adjective forms
                variations[word + 'er'] = score
                variations[word + 'est'] = score
            self.emotion_words[emotion].update(variations)

    def preprocess_text(self, text):
        """Preprocess input text"""
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

        # Split into words
        words = text.split()

        return words

    def train_model(self, csv_path='new_dataset.csv'):
        """Train a model based on the input CSV."""
        df = pd.read_csv(csv_path)
        df['emotion'] = df['emotion'].astype(str)  # Ensure emotion column is string for mapping

        X_train, X_test, y_train, y_test = train_test_split(
            df['sentence'], df['emotion'], test_size=0.2, random_state=42
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

    def predict_emotion(self, sentence):
        """Predict emotion for a given sentence."""
        sentence_vec = self.vectorizer.transform([sentence])
        emotion_code = self.model.predict(sentence_vec)[0]
        return self.emotion_mapping[emotion_code]



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
    detector.train_model('new_dataset.csv')  # Train on provided dataset

    # Add TF-IDF vectorization
    detector.add_tfidf()
    detector.train_model('new_dataset.csv')

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