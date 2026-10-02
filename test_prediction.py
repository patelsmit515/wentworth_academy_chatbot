import pandas as pd

from src.data_loader import load_intents
from src.features import add_text_features
from src.training import train_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load dataset
df = load_intents("data/intents.csv")

# Add engineered features
df = add_text_features(df)


# Train model
(
    model,
    preprocessor,
    X_train_processed,
    X_test_processed,
    y_test,
    X_test
) = train_model(df)


# New student messages
messages = [
    "Where can I find the library?",
    "Who teaches chemistry?",
    "When is my math test?",
    "Hi there",
    "I need help with algebra",
    "How do I register for classes and where is the registration office located",
    "The brown fox jumps over the lazy dog",
    "How are you doing",
    "I'm looking for the building where experiments are done!",
    "Can you tell me what comes next?",
    "Who do I report to",
    "Are students allowed to leave during the day?",
    "How do I bake a chocolate cake?",
    "What is the capital of France?",
    "I like playing football on weekends",
    "The weather is really nice today",
    "Can you help me fix my computer?",
    "Why is the ocean blue?"
]


# Create DataFrame for new messages
new_messages = pd.DataFrame({
    "text": messages
})


# Add the same engineered features
new_messages = add_text_features(new_messages)


# Transform using the already-trained preprocessor
new_messages_processed = preprocessor.transform(
    new_messages
)


# Predict
predictions = model.predict(
    new_messages_processed
)

probabilities = model.predict_proba(
    new_messages_processed
)


# Display predictions
print("\nPredictions:")
print("-----------------------")

for message, prediction, probability in zip(
    messages,
    predictions,
    probabilities
):
    confidence = probability.max()

    print(
        f"{message} → {prediction} "
        f"(confidence: {confidence:.2f})"
    )


# Test-set evaluation
test_predictions = model.predict(
    X_test_processed
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(f"\nAccuracy: {accuracy:.2f}")


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# Confusion matrix
labels = sorted(
    df["intent"].unique()
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(matrix)

print("\nLabels:")
print(labels)


# Misclassified examples
print("\n-----------------------")
print("Misclassified Examples:")
print("-----------------------")

for message, actual, predicted in zip(
    X_test["text"],
    y_test,
    test_predictions
):

    if actual != predicted:

        print(f"Message:   {message}")
        print(f"Actual:    {actual}")
        print(f"Predicted: {predicted}")
        print()