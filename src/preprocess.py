from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)   # unigrams + bigrams
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
