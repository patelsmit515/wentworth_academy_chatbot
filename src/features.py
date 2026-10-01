def create_text_features(texts):
    features = []

    for text in texts:
        words = text.split()

        word_count = len(words)
        char_count = len(text)
        avg_word_length = (
            sum(len(word) for word in words) / word_count
            if word_count > 0
            else 0
        )

        question_mark = int("?" in text)
        exclamation_mark = int("!" in text)

        features.append([
            word_count,
            char_count,
            avg_word_length,
            question_mark,
            exclamation_mark
        ])

    return features