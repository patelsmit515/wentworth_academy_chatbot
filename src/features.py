def add_text_features(df):
    df = df.copy()

    df["word_count"] = df["text"].apply(
        lambda text: len(text.split())
    )

    df["char_count"] = df["text"].apply(
        len
    )

    df["avg_word_length"] = df.apply(
        lambda row: (
            row["char_count"] / row["word_count"]
            if row["word_count"] > 0
            else 0
        ),
        axis=1
    )

    df["question_mark"] = df["text"].apply(
        lambda text: int("?" in text)
    )

    df["exclamation_mark"] = df["text"].apply(
        lambda text: int("!" in text)
    )

    return df