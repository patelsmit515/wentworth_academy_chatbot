from src.data_loader import load_intents
from src.training import train_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# Load dataset
df = load_intents("data/intents.csv")
unknown_df = pd.read_csv("data/unknown_examples.csv")

combined_df = pd.concat(
    [df, unknown_df],
    ignore_index=True
)

print("\nCombined dataset:")
print(f"Total examples: {len(combined_df)}")

print("\nIntent distribution:")
print(combined_df["intent"].value_counts())


# Train model
(
    model,
    vectorizer,
    X_train_tfidf,
    X_test_tfidf,
    y_test,
    X_test_text
) = train_model(combined_df)


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


# Convert messages into TF-IDF
messages_tfidf = vectorizer.transform(messages)


# Predict intents
predictions = model.predict(messages_tfidf)

# Get prediction probabilities
probabilities = model.predict_proba(messages_tfidf)


# Display predictions
for message, prediction, probability in zip(
    messages,
    predictions,
    probabilities
):
    confidence = probability.max()

    message_tfidf = vectorizer.transform([message])

    similarities = cosine_similarity(
        message_tfidf,
        X_train_tfidf
    )

    max_similarity = similarities.max()

    print(
        f"{message} → {prediction} "
        f"(confidence: {confidence:.2f}, "
        f"similarity: {max_similarity:.2f})"
    )


# Evaluate model on test data
test_predictions = model.predict(X_test_tfidf)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(f"\nAccuracy: {accuracy:.2f}")


# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# Confusion Matrix
labels = sorted(combined_df["intent"].unique())

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
print("-----------------------")
print("\nMisclassified Examples:")
print("-----------------------")

for message, actual, predicted in zip(
    X_test_text,
    y_test,
    test_predictions
):
    if actual != predicted:
        print(f"Message:   {message}")
        print(f"Actual:    {actual}")
        print(f"Predicted: {predicted}")
        print()

print("\nRelevant Training Examples:")
print("----------------------------")

for intent in [
    "club_information",
    "class_schedule",
    "teacher_information",
    "goodbye"
]:
    print(f"\n[{intent}]")

    examples = combined_df[
        combined_df["intent"] == intent
    ]["text"].tolist()

    for example in examples:
        print("-", example)
