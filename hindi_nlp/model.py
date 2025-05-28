# this is the code for our project
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
import re
import matplotlib.pyplot as plt

# Emotion dictionaries
emotion_words = {
       "happy":  {
        "खुश": 0.9, "मुस्कान": 0.75, "प्रसन्न": 1.0, "संतुष्ट": 0.85, "आनंदित": 0.95,
        "शांत": 0.85, "खुशी": 0.9, "सुखी": 0.95, "हर्षित": 0.9, "प्रसन्नचित्त": 1.0,
        "रोमांचित": 0.85, "मलंग": 0.8, "चमक": 0.9, "प्रसन्नता": 0.85, "मस्त": 1.0,
        "खिलखिलाना": 0.9, "आनंद": 0.85, "विभोर": 0.9, "हंसना": 0.95, "उल्लासित": 0.95,
        "हंसमुख": 0.9, "प्रफुल्लित": 1.0, "खिलखिलाहट": 0.95, "फूलों की तरह": 0.85, "चहकना": 0.9,
        "आनंदमय": 0.9, "उमंग": 0.95, "उत्साह": 0.9, "विजयी": 1.0, "मुस्कुराना": 0.9,
        "उमंगित": 0.95, "समाधान": 0.9, "चहक": 0.8, "खुशनुमा": 0.95, "संतोषजनक": 0.9,
        "राहत": 0.85, "अभिभूत": 0.9, "आशीर्वादित": 0.9, "रोमांच": 0.9, "संतोष": 0.95,
        "स्वस्थ": 0.85, "उत्साही": 0.95, "विस्मित": 0.8, "प्रसन्नतापूर्वक": 0.9, "शांति": 0.95,
        "सकारात्मक": 0.9, "हर्ष": 0.95, "चंचल": 0.9, "खुशदिल": 0.95, "आनंदितचित्त": 1.0
    },
    "joy": {
        "आनंद": 0.95, "मज़ा": 1.0, "उत्साहित": 0.95, "आनंदित": 0.95, "खुशी": 1.0,
        "हर्ष": 0.95, "आनंदमय": 0.95, "जश्न": 0.95, "उल्लास": 1.0, "जगमगाहट": 0.75,
        "सुख": 0.95, "विजय": 0.95, "खुशी": 1.0, "गर्व": 0.9, "मज़ेदार": 0.95,
        "शुभ": 0.8, "उत्सव": 1.0, "प्रसन्नता": 0.95, "चहलपहल": 0.9, "रोमांच": 0.9,
        "मस्ती": 1.0, "जश्न मनाना": 0.95, "आनंदित होना": 0.95, "विजयीभाव": 0.95, "मनमोहक": 0.95,
        "उल्लासित": 0.95, "प्रफुल्लित": 1.0, "धूमधाम": 0.75, "खुशमिजाज": 0.9, "हर्षोल्लास": 1.0,
        "सुखद": 0.9, "महिमा": 0.8, "आश्चर्यजनक": 0.95, "विजयोल्लास": 0.95, "आनंदमग्न": 0.95,
        "झूमना": 1.0, "गर्वित": 0.9, "आनंदितजनक": 0.95, "प्रफुल्लता": 1.0, "विजयी": 0.95,
        "प्रसन्नचित": 1.0, "गदगद": 0.8, "संतुष्टि": 0.9, "गौरवान्वित": 0.9, "उमंगित": 0.95,
        "मंगल": 1.0, "आनंदकारी": 0.9, "उत्साहजनक": 0.95, "विजयमय": 1.0, "मिलनसार": 0.95
    },
    "sad": {
        "दुखी": 0.85, "अवसाद": 0.8, "उदास": 0.7, "निराश": 0.85, "खिन्न": 0.75,
        "विषाद": 0.8, "रोना": 0.85, "गम": 0.8, "कष्ट": 0.7, "असंतोष": 0.75,
        "निराशाजनक": 0.85, "अकेला": 0.8, "हतोत्साहित": 0.75, "ग़मगीन": 0.8, "विफलता": 0.7,
        "बिछड़ना": 0.7, "टूटना": 0.75, "शोक": 0.85, "अवसादग्रस्त": 0.8, "उदासी": 0.75,
        "असफलता": 0.75, "तनाव": 0.7, "वेदना": 0.8, "चिंता": 0.7, "अभाव": 0.75,
        "दर्द": 0.8, "बेजान": 0.7, "खालीपन": 0.75, "आंसू": 0.8, "शिकायत": 0.75,
        "अशांत": 0.7, "दुख": 0.75, "सिसकना": 0.85,"छिपा": 0.8, "तन्हा": 0.75, "असंतुलित": 0.7,
        "संताप": 0.8, "छोड़ना": 0.75, "व्यथित": 0.8, "मायूस": 0.85, "विनाश": 0.8,
        "तलाक": 0.75, "रोदन": 0.8, "संवेदनशील": 0.7, "निराशा": 0.85, "बेचैनी": 0.7
    },
    "anger": {
        "गुस्सा": 0.85, "क्रोधित": 0.9, "नाराज": 0.85, "आक्रोश": 0.9, "चिढ़": 0.8,
        "चिड़चिड़ा": 0.75, "घृणा": 0.9, "नफरत": 0.85, "आक्रामक": 0.75, "असहिष्णु": 0.8,
        "चिड़चिड़ाहट": 0.8, "द्वेष": 0.9, "उग्र": 0.85, "हिंसक": 0.9, "भड़ास": 0.75,
        "खुंदक": 0.85, "आक्रामकता": 0.8, "भयंकर": 0.9, "गुस्से": 0.8, "आत्मघाती": 0.75,
        "दर्दनाक": 0.85, "असहनीय": 0.8, "भड़कना": 0.75, "तड़पना": 0.8, "अपमान": 0.85,
        "बदला": 0.75, "फटकार": 0.8, "अस्वीकार": 0.75, "नफरत": 0.85, "गसस":0.8, "कठोरता": 0.8,
        "साहस": 0.75, "उग्रता": 0.9, "नाराजगी": 0.85, "हिंसात्मक": 0.8, "असहमति": 0.7,
        "विद्रोह": 0.9, "उत्पीड़न": 0.8, "चोट": 0.85, "झगड़ा": 0.75, "तनाव": 0.8
    },
    "suspense": {
        "रहस्यमय": 0.85, "संदेह": 0.8, "उत्सुक": 0.85, "रहस्य": 0.9, "गुप्त": 0.8,
        "शंका": 0.75, "कुतूहल": 0.85, "विचित्र": 0.9, "अज्ञात": 0.8, "अनिश्चितता": 0.75,
        "धुंधला": 0.85, "छिपा": 0.8, "अंधकार": 0.9, "अनसुलझा": 0.8, "अनुमान": 0.75,
        "संशय": 0.8, "भ्रम": 0.85, "अप्रत्याशित": 0.75, "रहस्योद्घाटन": 0.85, "चौंकाने वाला": 0.9,
        "अज्ञेय": 0.75, "सस्पेंस": 0.9, "शीतलता": 0.8, "अनदेखा": 0.75, "घबराहट": 0.8,
        "अधूरा": 0.75, "रहस्यमयी": 0.85, "अनुकूल": 0.8, "संदेहपूर्ण": 0.75
    }
}
intensifiers = {
    'बहुत': 1.5,   # Very
    'सचमुच': 1.5,  # Really
    'इतना': 1.5,   # So
    'काफी': 2.0,   # Quite
    'बेहद': 2.0,   # Extremely
    'पूर्णतया': 1.8,  # Completely
    'पूरी तरह': 1.8,  # Utterly
    'गहरा': 1.8,    # Deeply
    'अत्यधिक': 2.0, # Incredibly
    'बिलकुल': 2.0   # Absolutely
}
negators = {'नहीं', 'ना', 'मत', 'बिलकुल नहीं'}

# Load dataset
df = pd.read_excel('./updated_hindi.xlsx')  # Replace with your file path

# Load stopwords
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

# Updated preprocess_text function
def preprocess_text(text):
    # Lowercase the text, remove extra spaces, and split into words
    words = text.strip().split()
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words
# Detect emotion from text using dictionaries
def detect_emotion(text):
    words = preprocess_text(text)
    emotion_scores = defaultdict(float)
    negated = False
    multiplier = 1.0

    print(f"Processing words: {words}")
    for word in words:
        if word in negators:
            negated = True
            print(f"Negator detected: {word}")
            continue
        if word in intensifiers:
            multiplier = intensifiers[word]
            print(f"Intensifier detected: {word}, multiplier: {multiplier}")
            continue
        for emotion, word_dict in emotion_words.items():
            if word in word_dict:
                score = word_dict[word] * multiplier
                score = -score if negated else score
                emotion_scores[emotion] += score
                print(f" Score: {score}")
                negated = False
        multiplier = 1.0

    print(f"Final Emotion Scores: {emotion_scores}")
    if not emotion_scores:
        return 'happy'  # Default to 'happy' if no matches
    return max(emotion_scores, key=emotion_scores.get)


# Map emotion labels
emotion_mapping = {0: 'happy', 1: 'joy', 2: 'sad', 3: 'anger', 4: 'suspense'}
df['emotion'] = df['emotion'].map(emotion_mapping)

# Prepare features and labels
df['processed_text'] = df['sentence'].apply(preprocess_text)
X = df['processed_text'].apply(lambda x: ' '.join(x))
y = df['emotion']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Handle class imbalance
oversampler = RandomOverSampler(random_state=42)
X_train_vec, y_train = oversampler.fit_resample(X_train_vec, y_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Confusion matrix visualization
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, cmap="viridis")


# Interactive prediction
if __name__ == "__main__":
    while True:
        print("Enter a Hindi text to predict the emotion (or 'q' to quit):")
        user_input = input()
        if user_input.lower() == 'q':
            break

        # Preprocess input with stopword removal
        processed_input = ' '.join(preprocess_text(user_input))

        # Predict using the custom dictionary-based function
        predicted_emotion = detect_emotion(processed_input)

        # Ensure the emotion is valid before encoding
        if predicted_emotion in emotion_mapping.values():
            predicted_label = label_encoder.transform([predicted_emotion])[0]
            print(f"Predicted Emotion: {predicted_emotion}")
            print(f"Predicted Label (Encoded): {predicted_label}")
        else:
            print("Unable to predict emotion.")