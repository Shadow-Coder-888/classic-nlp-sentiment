import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_loader import load_data
from preprocess import vectorize_text

def train():
    texts, labels = load_data("data/dataset.csv")

    X, vectorizer = vectorize_text(texts)

    model = MultinomialNB()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy")

    print("Cross-validation accuracies:", scores)
    print("Mean accuracy:", scores.mean())

    # Train final model on full dataset
    model.fit(X, labels)

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

if __name__ == "__main__":
    train()
