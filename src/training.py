
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.preprocessing import create_vectorizer


def train_model(df):
    X = df["text"]
    y = df["intent"]

    # Convert intent names into numbers for XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    # Create TF-IDF vectorizer
    vectorizer = create_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Convert test labels back to intent names
    y_test_labels = label_encoder.inverse_transform(y_test)

    return (
        model,
        vectorizer,
        X_train_tfidf,
        X_test_tfidf,
        y_test_labels,
        X_test,
        label_encoder
    )   

