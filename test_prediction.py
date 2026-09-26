from src.data_loader import load_intents
from src.training import train_model

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
    "how do you sleep at night knowing that you are a terrible person",
]

# Convert messages into TF-IDF
messages_tfidf = vectorizer.transform(messages)

# Predict intents
predictions = model.predict(messages_tfidf)

for message, prediction in zip(messages, predictions):
    print(f"'{message}' → {prediction}")