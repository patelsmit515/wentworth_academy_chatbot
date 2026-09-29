from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.preprocessing import create_vectorizer

def train_model(df):
    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    vectorizer = create_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, X_test_tfidf, y_test, X_test