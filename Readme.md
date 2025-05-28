# NLP_Final_Project

This project contains a comprehensive emotion detection system divided into English and Hindi components, integrating both NLP and ML approaches. The structure includes three main sub-projects: `hindi_nlp`, `english_nlp`, and `english_nlp_ml`.

## Project Structure

```
NLP_final_project/
│
├── hindi_nlp/
│   ├── preprocess.py
│   ├── lemma.txt
│   ├── model.py
│   ├── updated_hindi.xlsx
│   └── stopwords.txt
│
├── english_nlp/
│   ├── data_processing.py
│   ├── shuffled.csv
│   ├── lemmatization.txt
│   ├── graph.py
│   ├── FinalData.csv
│   └── stopwords.txt
│   └── web_scrap.py
│
└── english_nlp_ml/
    ├── NLP_ML_hybrid.py
    ├── new_dataset.csv
```

## Overview of Sub-Projects

## Overview

English:
The system performs the following operations:

1. Scrapes comments from YouTube videos
2. Processes the comments for emotion analysis
3. Detects emotions using a pure NLP approach
4. Classifies emotions into six categories:
   - Sad (0)
   - Happy (1)
   - Love (2)
   - Anger (3)
   - Fear (4)
   - Surprise (5)

Hindi:
The system performs the following operations:

1. Scrapes and collects sentences from various hindi textbooks and websites.
2. Processes the sentences for emotion analysis
3. Detects emotions using a pure NLP approach
4. Classifies emotions into six categories:
   - Happy (0)
   - Joy (1)
   - Sad (2)
   - Anger (3)
   - Suspense (4)

## Prerequisites

- Python 3.7+
- Required Python packages:

  pandas
  requests
  langdetect
  nltk
  scikit-learn [ for calculating precision, accuracy etc]

## Usage

### 1. Scraping YouTube Comments

python webScrape.py

This will:

- Fetch comments from the specified YouTube video
- Filter for English comments
- Save raw comments to `Scrapped.csv`

Configuration options:

- `video_id`: YouTube video ID to scrape
- `max_comments`: Maximum number of comments to fetch (default: 100)

### 2. Emotion Detection(English and Hindi)

python emotion_detector.py
python model.py

This will:

- Load the preprocessed dataset
- Train the emotion detection model
- Process the comments / sentences in hindi
- Generate emotion predictions
- Save results to `FinalData.csv`/ `Output_processed.xlsx`

## Emotion Detection Model

The model uses:

- Word-based emotion dictionaries
- Bigram analysis
- Intensity multipliers
- Negation handling
- Lemmatization

#### Key Features:

- **Preprocessing**: Tokenization, lemmatization, and stopword removal.
- **Emotion Classification**: A mix of pure NLP approaches with rule-based analysis.
- **Graphical Representation**: `graph.py` visualizes the detected emotions.

#### Techniques Used:

- Text normalization and cleaning.
- Rule-based emotion detection.
- Data visualization with `matplotlib`.

### 2. english_nlp: Pure NLP Model for Emotion Analysis

This module is designed to analyze emotions in English text using NLP methods, emphasizing pre-trained language resources and feature extraction.

#### Key Features:

- **Data Preprocessing**: Advanced text cleaning and tokenization.
- **Emotion Detection**: Utilizes linguistic rules and text-based heuristics.
- **Visualization**: Emotion distribution graphs using `graph.py`.

### 3. english_nlp_ml: NLP with Machine Learning

This module combines NLP techniques with machine learning algorithms for more sophisticated emotion detection.

#### Key Features:

- **Vectorization Methods**: Count vectorization and TF-IDF.
- **Model Training**: Logistic regression and support for various n-gram extractions.
- **Evaluation**: Model performance on test data with accuracy and confusion matrix.

#### Techniques Used:

- ML model implementation (`NLP_ML_hybrid.py`).
- Training with custom `new_dataset.csv`.
- End-to-end integration of NLP pre-processing and ML pipeline.

---

### Text Processing Features

English

- Contraction expansion
- Emoji removal
- URL removal
- Punctuation handling
- Stopword removal
- Lemmatization

Hindi

- hastag mention patterns,number_pattern and extra spaces
- Non hindi words/charcaters
- URL removal
- Punctuation handling
- Stopword removal
- Lemmatization

English

### Dataset.csv

sentence,emotion
"What was that noise? I'm too scared to investigate.",4
"I can't believe they messed up AGAIN!!",3

Hindi

### Dataset.csv

"वह अपने दोस्तों के साथ खेलकर प्रसन्न था|",1

English

### FinalData.csv

sentence,emotion,predicted_emotion
"what was that noise im too scared to investigate",4,4
"i cant believe they messed up again",3,3

Hindi

### Output.csv

वह अपने दोस्तों के साथ खेलकर प्रसन्न था
Processing words: ['दोस्तों', 'खेलकर', 'प्रसन्न']
Predicted Emotion: happy
Predicted Label (Encoded): 1

## Performance

The model's accuracy is calculated by comparing predicted emotions with labeled data. Current performance metrics are displayed after processing the dataset.

---------------------- > next model

# NLP_ML_Hybrid: Emotion Detection System

## Overview

`NLP_ML_hybrid.py` is a Python script for building a comprehensive emotion detection system. This program uses machine learning (ML) and natural language processing (NLP) techniques to classify sentences into emotions such as _happy_, _sad_, _love_, _anger_, _fear_, and _surprise_. The model is trained using a logistic regression classifier with support for count vectorization, TF-IDF vectorization, and n-gram extraction (unigrams, bigrams, and trigrams).

## Features

- Preprocessing of text: normalization, contraction handling, and punctuation removal.
- Training of logistic regression with options for:
  - CountVectorizer for unigram and n-gram analysis.
  - TF-IDF vectorization.
  - Customizable n-gram ranges (unigrams to trigrams).
- Comprehensive emotion classification report with accuracy measurement.
- Interactive console-based prediction system for user input.

## Setup Instructions

Follow these steps to set up the project and run the code.

### 1. Create and Activate a Virtual Environment

Ensure Python is installed on your system. Create and activate a virtual environment to keep your project dependencies organized.

**Windows:**

```bash
python -m venv venv
./venv/scripts/activate

python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Packages

Navigate to the project folder and install the necessary Python packages using pip

```bash
cd NLP_project
pip install pandas scikit-learn
```

### Place Your Dataset

Ensure you have the dataset new_dataset.csv in the NLP_project folder. This dataset should contain columns with sentence and emotion data for training.

## ML and NLP Techniques Used

### Machine Learning Techniques

#### Logistic Regression

A simple yet effective classification model used for predicting categorical outcomes.

#### Model Training and Evaluation

Includes training with data splitting (train_test_split) and model performance evaluation with classification_report and accuracy_score

### Natural Language Processing Techniques

#### Text Preprocessing:

      Lowercasing of text for normalization.
      Contraction expansion (e.g., "don't" to "do not").
      Basic punctuation removal

#### Feature Extraction:

      CountVectorizer: Converts text into a vector of word frequencies.
      TF-IDF Vectorizer: Assigns weights to words based on term frequency and inverse document frequency.
      N-grams: Adds context by considering sequences of words (unigrams, bigrams, and trigrams).

### Custom Enhancements

      Emotion Dictionary Initialization: Handcrafted word lists for emotions to support model learning.

      Word Variations: Automatic generation of common word forms (e.g., adding 'ing', 'ed', 's').

### Example Output

```bash
Enter a sentence to analyze: I am so happy today!

Predicted Emotion: HAPPY
```
