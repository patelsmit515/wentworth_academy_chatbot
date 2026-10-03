from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.preprocessing import create_preprocessor


def train_model(df):

    X = df[
        [
            "text",
            "word_count",
            "char_count",
            "avg_word_length",
            "question_mark",
            "exclamation_mark",
            "question_signal",
            "greeting_signal",
            "goodbye_signal",
            "exam_signal",
            "teacher_signal"
        ]
    ]

    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Create preprocessing system
    preprocessor = create_preprocessor()

    # Learn transformations from training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same transformations to test data
    X_test_processed = preprocessor.transform(X_test)

    # Create Logistic Regression model
    model = LogisticRegression(
        C=10,
        max_iter=1000
    )

    # Train model
    model.fit(
        X_train_processed,
        y_train
    )

    return (
        model,
        preprocessor,
        X_train_processed,
        X_test_processed,
        y_test,
        X_test
    )