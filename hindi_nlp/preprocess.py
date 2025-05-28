import re
import pandas as pd


_non_hindi = re.compile(r'[^\u0900-\u097F\u200C\u200D\s]', flags=re.UNICODE)  
_url_pattern = re.compile(r'http\S+|www\S+')
_hashtag_mention_pattern = re.compile(r'[@#]\w+')
_number_pattern = re.compile(r'\d+')
_extra_spaces = re.compile(r'\s+')
_punctuation_pattern = re.compile(r'[^\w\s\u0900-\u097F]')  


def load_stopwords(filepath):
  
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        swords = f.read().split('\n')
        if swords[-1] == '':
            swords.pop()
    return swords


def load_lemmatization(filepath):
  
    lemma_dict = {}
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or len(line.split()) != 2:
                continue
            word, lemma = line.split()
            lemma_dict[word] = lemma
    return lemma_dict

def clean_text(data):
   
    data = _url_pattern.sub("", data)
    data = _hashtag_mention_pattern.sub("", data)
    data = _number_pattern.sub("", data)
    data = _non_hindi.sub("", data) 
    data = _punctuation_pattern.sub("", data)  
    data = _extra_spaces.sub(" ", data)
    return data.strip()

def remove_stopwords_and_lemmatize(data, swords, lemma_dict):
   
    data = clean_text(data)

    data = data.split()


    clean = [
        lemma_dict.get(word, word) 
        for word in data if len(word) >= 3 and word not in swords
    ]

    return ' '.join(clean)

def pre_process(data, swords, lemma_dict):
   
    data = remove_stopwords_and_lemmatize(data, swords, lemma_dict)
    return data

def process_excel_file(input_file, output_file, stopwords_file, lemma_file, sheet_name=0, column_name='Sentences'):
    
    swords = load_stopwords(stopwords_file)
    lemma_dict = load_lemmatization(lemma_file)
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print("Columns in the Excel file:", df.columns)

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")

    df['processed_text'] = df[column_name].apply(lambda x: pre_process(str(x), swords, lemma_dict))

    df.to_excel(output_file, index=False)


process_excel_file('updated_hindi.xlsx', 'output_processed.xlsx', 'stopwords.txt', 'lemma.txt', sheet_name=0, column_name='Sentences')
