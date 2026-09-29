from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():
    vectorizer = TfidfVectorizer(ngram_range = (1,2))

    return vectorizer