from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.preprocessing import create_preprocessor


def train_mlp(df):

    X = df[
        [
            "text",
            "word_count",
            "char_count",
            "avg_word_length",
            "question_mark",
            "exclamation_mark",
            "exam_signal"
        ]
    ]

    y = df["intent"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Create preprocessing
    preprocessor = create_preprocessor()

    # Fit preprocessing on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Neural network
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    # Train
    model.fit(
        X_train_processed,
        y_train
    )

    return (
        model,
        preprocessor,
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        X_test
    )