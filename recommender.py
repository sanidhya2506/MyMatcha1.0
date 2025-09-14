import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk
from difflib import get_close_matches  # for fuzzy matching

# Ensure nltk punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

def load_and_preprocess_data():
    # Replace this URL with your Google Sheet CSV export link
    sheet_url = "https://docs.google.com/spreadsheets/d/1-bHukazIbC7jwghhwUuQO50rtEqGA27JgLfl5jK03jk/export?format=csv"
    df = pd.read_csv(sheet_url)

    df['Episodes'] = df['Episodes'].fillna(12)
    df['Duration'] = df['Duration'].fillna("45 min")
    df['Rating'] = df['Rating'].fillna("5.7/10 (MDL)")
    df['Release Date'] = df['Release Date'].fillna("2021-01-01")

    df.drop_duplicates(subset='Title', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Strip extra spaces from titles
    df['Title'] = df['Title'].str.strip()

    df['Themes'] = df['Themes'].apply(lambda x: x.replace('-', ' ') if isinstance(x, str) else x)
    df['Tags'] = df.apply(lambda row: f"{row['Summary']} {row['Genres']} {row['Themes']}".lower(), axis=1)

    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])

    newdf = df[['Title', 'Country', 'Rating', 'Tags', 'Poster_URL']].copy()
    newdf['Tags'] = newdf['Tags'].apply(stem)

    cv = CountVectorizer(max_features=len(newdf), stop_words='english')
    vectors = cv.fit_transform(newdf['Tags']).toarray()
    similarity = cosine_similarity(vectors)

    return newdf, similarity

# Load data once
newdf, similarity = load_and_preprocess_data()

def recommend(drama_title):
    drama_title = drama_title.strip().lower()
    
    # Fuzzy match to find closest title
    titles_lower = [t.lower() for t in newdf['Title'].tolist()]
    closest_matches = get_close_matches(drama_title, titles_lower, n=1, cutoff=0.6)
    
    if not closest_matches:
        return []  # no close match found
    
    # Find index of closest match
    drama_index = titles_lower.index(closest_matches[0])
    
    distances = similarity[drama_index]
    drama_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [newdf.iloc[i[0]].Title for i in drama_list]

# Flask expects this function name
def get_recommendations(title):
    return recommend(title)
