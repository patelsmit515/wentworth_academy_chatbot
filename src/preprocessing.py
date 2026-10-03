from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


TEXT_COLUMN = "text"

NUMERIC_FEATURES = [
    "word_count",
    "char_count",
    "avg_word_length",
    "question_mark",
    "exclamation_mark",
    "exam_signal"
]


def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1)
                ),
                TEXT_COLUMN
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ]
    )

    return preprocessor

