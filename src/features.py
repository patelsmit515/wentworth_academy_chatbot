def add_text_features(df):

    df = df.copy()

    # Basic text features
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

    # Targeted semantic features
    df["question_signal"] = df["text"].apply(
        lambda text: int(
            any(
                word in text.lower().split()
                for word in [
                    "what",
                    "when",
                    "where",
                    "who",
                    "which",
                    "how",
                    "why"
                ]
            )
        )
    )

    df["greeting_signal"] = df["text"].apply(
        lambda text: int(
            any(
                phrase in text.lower()
                for phrase in [
                    "hello",
                    "hi",
                    "hey",
                    "good morning",
                    "good afternoon",
                    "good evening",
                    "good to see you",
                    "nice to see you"
                ]
            )
        )
    )

    df["goodbye_signal"] = df["text"].apply(
        lambda text: int(
            any(
                phrase in text.lower()
                for phrase in [
                    "bye",
                    "goodbye",
                    "leave",
                    "leaving",
                    "heading out",
                    "see you",
                    "gotta go",
                    "have to go",
                    "need to go"
                ]
            )
        )
    )

    df["exam_signal"] = df["text"].apply(
        lambda text: int(
            any(
                word in text.lower().split()
                for word in [
                    "exam",
                    "test",
                    "tested",
                    "testing",
                    "quiz",
                    "assessment",
                    "midterm",
                    "final"
                ]
            )
        )
    )

    df["teacher_signal"] = df["text"].apply(
        lambda text: int(
            any(
                phrase in text.lower()
                for phrase in [
                    "teacher",
                    "professor",
                    "instructor",
                    "sir",
                    "ma'am",
                    "who teaches",
                    "which teacher"
                ]
            )
        )
    )

    return df