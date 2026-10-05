import pandas as pd

from src.features import add_text_features


def predict_intent(
    message,
    model,
    preprocessor
):

    # Simple goodbye detection
    goodbye_phrases = [
        "goodbye",
        "good bye",
        "bye",
        "see you",
        "talk to you later",
        "i have to go",
        "i need to go",
        "i'm leaving",
        "i am leaving"
    ]

    message_lower = message.lower()

    if any(
        phrase in message_lower
        for phrase in goodbye_phrases
    ):
        return "goodbye", 1.00


    # Create DataFrame for the new message
    new_message = pd.DataFrame({
        "text": [message]
    })

    # Add the same features used during training
    new_message = add_text_features(new_message)

    # Apply the trained preprocessor
    message_processed = preprocessor.transform(
        new_message
    )

    # Predict intent
    prediction = model.predict(
        message_processed
    )[0]

    # Get prediction confidence
    probabilities = model.predict_proba(
        message_processed
    )[0]

    confidence = probabilities.max()

    return prediction, confidence