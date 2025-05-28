import re
from collections import defaultdict
import pandas as pd
import string
from nltk.util import ngrams


emotion_labels = {"sad": 0, "happy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}


class EmotionDetector:
    def __init__(self, lemmatization_dict, stopwords):
        self.initialize_dictionaries()
        self.lemmatization_dict = lemmatization_dict
        self.stopwords = stopwords
        self.initialize_bigram_dictionaries()  # Add this line

    def initialize_bigram_dictionaries(self):

        self.emotion_bigrams = {
            "happy": {
                ("very", "happy"): 1.5,
                ("so", "happy"): 1.5,
                ("really", "happy"): 1.5,
                ("feeling", "good"): 1.2,
                ("feeling", "great"): 1.3,
                ("pure", "joy"): 1.4,
                ("absolutely", "wonderful"): 1.5,
                ("truly", "blessed"): 1.3,
                ("incredibly", "happy"): 1.5,
                ("perfectly", "content"): 1.2,
                ("extremely", "pleased"): 1.4,
                ("totally", "awesome"): 1.3,
            },
            "sad": {
                ("very", "sad"): 1.5,
                ("so", "sad"): 1.5,
                ("really", "sad"): 1.5,
                ("deeply", "hurt"): 1.4,
                ("heart", "broken"): 1.5,
                ("absolutely", "devastated"): 1.6,
                ("feeling", "down"): 1.2,
                ("truly", "sorry"): 1.3,
                ("incredibly", "disappointed"): 1.4,
                ("totally", "heartbroken"): 1.5,
                ("completely", "devastated"): 1.6,
            },
            "anger": {
                ("very", "angry"): 1.5,
                ("so", "angry"): 1.5,
                ("really", "angry"): 1.5,
                ("absolutely", "furious"): 1.6,
                ("extremely", "mad"): 1.5,
                ("totally", "frustrated"): 1.4,
                ("completely", "outraged"): 1.6,
                ("deeply", "annoyed"): 1.3,
                ("truly", "frustrated"): 1.4,
                ("incredibly", "angry"): 1.5,
            },
            "fear": {
                ("very", "scared"): 1.5,
                ("so", "afraid"): 1.5,
                ("really", "terrified"): 1.5,
                ("absolutely", "terrified"): 1.6,
                ("completely", "frightened"): 1.5,
                ("truly", "scared"): 1.4,
                ("deeply", "worried"): 1.3,
                ("extremely", "anxious"): 1.5,
                ("totally", "paranoid"): 1.4,
                ("incredibly", "nervous"): 1.4,
            },
            "surprise": {
                ("very", "surprised"): 1.5,
                ("so", "surprised"): 1.5,
                ("really", "surprised"): 1.5,
                ("absolutely", "amazed"): 1.6,
                ("completely", "shocked"): 1.5,
                ("totally", "unexpected"): 1.4,
                ("truly", "astonished"): 1.5,
                ("incredibly", "surprised"): 1.5,
                ("deeply", "shocked"): 1.4,
                ("utterly", "amazed"): 1.6,
            },
            "love": {
                ("deeply", "love"): 1.5,
                ("truly", "love"): 1.5,
                ("really", "love"): 1.4,
                ("absolutely", "adore"): 1.6,
                ("completely", "devoted"): 1.5,
                ("totally", "love"): 1.4,
                ("truly", "cherish"): 1.5,
                ("deeply", "cherish"): 1.5,
                ("incredibly", "romantic"): 1.4,
                ("purely", "love"): 1.5,
            },
        }

    def initialize_dictionaries(self):
        """Initialize emotion dictionaries with comprehensive word lists"""
        self.emotion_words = {
            "happy": {
                # Core happiness words
                "happy": 1.0,
                "joy": 1.0,
                "delighted": 0.9,
                "excited": 0.9,
                "wonderful": 0.9,
                "great": 0.8,
                "pleased": 0.8,
                "glad": 0.8,
                "fantastic": 0.9,
                "thrilled": 0.9,
                "enjoy": 0.8,
                "cheerful": 0.8,
                "blessed": 0.8,
                "smile": 0.7,
                "laugh": 0.8,
                "content": 0.7,
                "satisfied": 0.7,
                "peaceful": 0.6,
                "proud": 0.7,
                "fun": 0.7,
                "awesome": 0.9,
                "amazing": 0.9,
                "excellent": 0.8,
                "loving": 0.8,
                "thankful": 0.7,
                "grateful": 0.8,
                "bliss": 0.9,
                "joy": 0.9,
                "celebrate": 0.8,
                "win": 0.7,
                "success": 0.7,
                "perfect": 0.8,
                "beautiful": 0.7,
                "bright": 0.6,
            },
            "sad": {
                # Core sadness words
                "sad": 1.0,
                "unhappy": 0.9,
                "depressed": 0.9,
                "miserable": 0.9,
                "heartbroken": 1.0,
                "disappointed": 0.8,
                "upset": 0.8,
                "hurt": 0.8,
                "crying": 0.9,
                "cry": 0.8,
                "tears": 0.8,
                "grief": 0.9,
                "grieving": 0.9,
                "sorrow": 0.9,
                "pain": 0.8,
                "painful": 0.8,
                "alone": 0.7,
                "lonely": 0.8,
                "lost": 0.7,
                "broken": 0.8,
                "sorry": 0.6,
                "regret": 0.7,
                "hopeless": 0.9,
                "despair": 0.9,
                "fail": 0.7,
                "failed": 0.7,
                "failure": 0.7,
                "miss": 0.6,
                "missing": 0.6,
                "empty": 0.8,
            },
            "anger": {
                # Core anger words
                "angry": 1.0,
                "mad": 0.9,
                "furious": 1.0,
                "rage": 1.0,
                "hate": 0.9,
                "frustrate": 0.8,
                "frustrated": 0.8,
                "irritation": 0.7,
                "annoyed": 0.7,
                "upset": 0.7,
                "bitter": 0.8,
                "disgusted": 0.8,
                "disgusting": 0.8,
                "hostile": 0.9,
                "anger": 1.0,
                "outraged": 0.9,
                "offensive": 0.7,
                "aggressive": 0.8,
                "fury": 1.0,
                "wrathful": 0.9,
                "threatening": 0.8,
                "violent": 0.9,
                "destroy": 0.8,
                "revenge": 0.8,
                "cruel": 0.8,
            },
            "fear": {
                # Core fear words
                "afraid": 1.0,
                "scared": 1.0,
                "frightened": 1.0,
                "terrified": 1.0,
                "panic": 0.9,
                "fear": 1.0,
                "worried": 0.8,
                "worry": 0.8,
                "anxious": 0.8,
                "nervous": 0.8,
                "horrified": 0.9,
                "horror": 0.9,
                "terrible": 0.8,
                "terror": 1.0,
                "scary": 0.8,
                "scared": 1.0,
                "frightening": 0.9,
                "afraid": 1.0,
                "dread": 0.9,
                "stressed": 0.7,
                "stress": 0.7,
                "paranoid": 0.8,
                "suspicious": 0.6,
                "doubt": 0.6,
                "uneasy": 0.7,
            },
            "surprise": {
                # Core surprise words
                "surprised": 1.0,
                "wow": 0.9,
                "amazing": 0.8,
                "astonished": 1.0,
                "shocked": 0.9,
                "unexpected": 0.8,
                "sudden": 0.7,
                "startled": 0.8,
                "surprise": 1.0,
                "stunning": 0.8,
                "incredible": 0.8,
                "unbelievable": 0.9,
                "wonder": 0.7,
                "wonderful": 0.7,
                "magical": 0.7,
                "miraculous": 0.8,
                "spectacular": 0.8,
                "amazed": 0.9,
                "extraordinary": 0.7,
                "odd": 0.6,
            },
            "love": {
                # Core love words
                "love": 1.0,
                "adore": 0.9,
                "cherish": 0.9,
                "passionate": 0.8,
                "romantic": 0.8,
                "affectionate": 0.8,
                "caring": 0.7,
                "lovely": 0.7,
                "sweetest": 0.8,
                "darling": 0.8,
                "devoted": 0.9,
                "beloved": 0.9,
                "precious": 0.8,
                "intimate": 0.8,
                "close": 0.7,
                "loved": 1.0,
                "loves": 1.0,
                "loving": 0.9,
                "romance": 0.8,
                "beautiful": 0.7,
                "tender": 0.7,
                "gentle": 0.6,
                "warm": 0.6,
                "together": 0.6,
                "forever": 0.7,
            },
        }

        # Initialize intensifiers with their multipliers
        self.intensifiers = {
            "very": 1.5,
            "really": 1.5,
            "so": 1.5,
            "extremely": 2.0,
            "absolutely": 2.0,
            "completely": 1.8,
            "totally": 1.8,
            "utterly": 1.8,
            "deeply": 1.8,
            "incredibly": 2.0,
            "intensely": 1.8,
            "particularly": 1.3,
            "quite": 1.3,
            "rather": 1.2,
            "somewhat": 0.8,
        }

        # Initialize negators
        self.negators = {
            "not",
            "never",
            "no",
            "n't",
            "cannot",
            "cant",
            "wont",
            "wouldnt",
            "shouldnt",
            "couldnt",
        }

        # Add word variations to emotion dictionaries
        self._add_word_variations()

    def _add_word_variations(self):
        """Add common variations of emotional words"""
        for emotion, words in self.emotion_words.items():
            variations = {}
            for word, score in words.items():
                # Add common verb forms
                variations[word + "ing"] = score
                variations[word + "ed"] = score
                variations[word + "s"] = score

                # Add common adjective forms
                variations[word + "er"] = score
                variations[word + "est"] = score
                # Add plural forms
                if word[-1] != "s":
                    variations[word + "s"] = score
            self.emotion_words[emotion].update(variations)

    def get_bigrams(self, words):
        """Generate bigrams from a list of words"""
        return list(ngrams(words, 2))

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
        text = re.sub(r"[^\w\s\']", " ", text)

        # Split into words
        words = text.split()

        return words

    def detect_emotion(self, text):
        words = self.preprocess_text(text)
        emotion_scores = defaultdict(float)

        # Process bigrams
        bigrams = self.get_bigrams(words)
        for bigram in bigrams:
            for emotion, bigram_dict in self.emotion_bigrams.items():
                if bigram in bigram_dict:
                    emotion_scores[emotion] += bigram_dict[bigram]

        # Process individual words
        i = 0
        while i < len(words):
            word = words[i]

            # Initialize multiplier for intensity
            multiplier = 1.0

            # Check for intensifiers
            if i > 0 and words[i - 1] in self.intensifiers:
                multiplier = self.intensifiers[words[i - 1]]

            # Check for negators
            negated = False
            if i > 0 and words[i - 1] in self.negators:
                negated = True

            # Update emotion scores
            for emotion, emotion_dict in self.emotion_words.items():
                if word in emotion_dict:
                    score = emotion_dict[word] * multiplier
                    if negated:
                        score = -score
                    emotion_scores[emotion] += score

            i += 1

        # Handle empty scores
        if not emotion_scores:
            return 1, {}  # Default: 'happy' emotion detected

        # Normalize scores
        max_score = max(emotion_scores.values())
        if max_score > 0:
            emotion_scores = {k: v / max_score for k, v in emotion_scores.items()}

        # Get primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        predicted_label = emotion_labels.get(primary_emotion, -1)

        return predicted_label, emotion_scores


# [Rest of the code remains the same]


def load_lemmatization_dict(file_path):
    lemmatization_dict = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if "\t" in line:
                    key, value = line.split("\t", 1)
                    lemmatization_dict[key.strip()] = value.strip()
    except Exception as e:
        raise RuntimeError(f"Error loading lemmatization dictionary: {e}")
    return lemmatization_dict


def load_stopwords(file_path):
    stopwords = set()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                stopwords.add(line.strip())
    except Exception as e:
        raise RuntimeError(f"Error loading stopwords: {e}")
    return stopwords


def manual_lemmatize(word, lemmatization_dict):
    return lemmatization_dict.get(word, word)


def expand_contractions(text):
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "didn't": "did not",
        "doesn't": "does not",
        "isn't": "is not",
        "it's": "it is",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "we're": "we are",
        "they're": "they are",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would",
        "aren't": "are not",
        "weren't": "were not",
        "can't've": "cannot have",
        "could've": "could have",
        "should've": "should have",
        "would've": "would have",
        "might've": "might have",
        "must've": "must have",
        "ain't": "am not",
        "isn't": "is not",
        "hasn't": "has not",
        "haven't": "have not",
        "wasn't": "was not",
        "weren't": "were not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "won't": "will not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text


def remove_emojis(text):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U00002700-\U000027BF]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def preprocess_text(text, lemmatization_dict, stopwords):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = remove_emojis(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # text = str(TextBlob(text).correct())
    text = " ".join(word for word in text.split() if word not in stopwords)
    text = " ".join(manual_lemmatize(word, lemmatization_dict) for word in text.split())

    return text


def load_and_preprocess(file_path, lemmatization_dict, stopwords, emotion_detector):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        if "sentence" in df.columns and "emotion" in df.columns:
            df["sentence"] = (
                df["sentence"]
                .astype(str)
                .apply(lambda x: preprocess_text(x, lemmatization_dict, stopwords))
            )
            predicted_emotions, _ = zip(
                *df["sentence"].apply(emotion_detector.detect_emotion)
            )

            # Align predictions with DataFrame indices
            predicted_emotions_series = pd.Series(predicted_emotions, index=df.index)

            # Safely compare and assign predicted_emotion
            df["predicted_emotion"] = df["emotion"].where(
                df["emotion"] == predicted_emotions_series, 0
            )
        else:
            raise ValueError("Required columns 'sentence' and 'emotion' not found.")

        return df
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the CSV: {e}")


def save_to_csv(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"CSV file saved as: {output_file}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving to CSV: {e}")


if __name__ == "__main__":
    input_file = "./Dataset.csv"
    output_file = "./FinalData.csv"
    lemmatization_dict_file = "./lemmatization.txt"
    stopwords_file = "./stopwords.txt"

    try:
        lemmatization_dict = load_lemmatization_dict(lemmatization_dict_file)
        stopwords = load_stopwords(stopwords_file)

        emotion_detector = EmotionDetector(lemmatization_dict, stopwords)
        preprocessed_df = load_and_preprocess(
            input_file, lemmatization_dict, stopwords, emotion_detector
        )
        save_to_csv(preprocessed_df, output_file)

        true_labels = preprocessed_df["emotion"]
        predicted_labels = preprocessed_df["predicted_emotion"]

        n = len(true_labels)
        count = 0
        for index, row in preprocessed_df.iterrows():
            if row["emotion"] == row["predicted_emotion"]:
                count = count + 1
        accuracy = count / n

        print(f"Accuracy: {accuracy:.2f}")

        while True:
            input_text = input("Enter text (or 'q' to quit): ")
            if input_text.lower() == "q":
                break
            emotion, _ = emotion_detector.detect_emotion(input_text)
            print(f"Predicted emotion: {list(emotion_labels.keys())[emotion]}")
    except Exception as e:
        print(e)
