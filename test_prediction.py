from src.data_loader import load_intents
from src.training import train_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = load_intents("data/intents.csv")

# Train model
model, vectorizer, X_test, y_test = train_model(df)

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
    "Are students allowed to leave during the day?"
]

# Convert messages into TF-IDF
messages_tfidf = vectorizer.transform(messages)

# Predict intents
predictions = model.predict(messages_tfidf)

for message, prediction in zip(messages, predictions):
    print(f"'{message}' → {prediction}")


# Evaluate model on test data
test_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

print("Model Evaluation")

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# Confusion Matrix
labels = sorted(df["intent"].unique())

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(matrix)

print("\nLabels:")
print(labels)
    
