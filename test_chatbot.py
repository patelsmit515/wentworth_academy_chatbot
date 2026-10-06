from src.data_loader import load_intents
from src.features import add_text_features
from src.training_mlp import train_mlp
from src.prediction import predict_intent
from src.response import get_response


# Load dataset
df = load_intents("data/intents.csv")

# Add features
df = add_text_features(df)


# Train MLP
(
    model,
    preprocessor,
    _,
    _,
    _,
    _,
    _
) = train_mlp(df)


# Test messages
messages = [
    "Hi there",
    "I need help with algebra",
    "Who teaches chemistry?",
    "Where can I find the library?",
    "What clubs can I join?",
    "Goodbye"
]


print("\nWentworth Academy Chatbot Test")
print("------------------------------")


for message in messages:

    intent, confidence = predict_intent(
        message,
        model,
        preprocessor
    )

    response = get_response(
        intent,
        message
    )

    print(f"\nStudent: {message}")
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bot: {response}")