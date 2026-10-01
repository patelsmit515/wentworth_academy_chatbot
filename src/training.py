from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Create TF-IDF vectorizer
    vectorizer = create_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000
    )

    # Hyperparameters to test
    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5, 10]
    }

    # Grid Search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(X_train_tfidf, y_train)

    # Get best model
    model = grid_search.best_estimator_

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    print(f"Best CV Accuracy: {grid_search.best_score_:.2f}")

    return (
        model,
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
        y_test,
        X_test
    )