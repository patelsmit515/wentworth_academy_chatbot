from src.data_loader import load_intents
from src.features import add_text_features
from src.training_mlp import train_mlp
from src.prediction import predict_intent


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
    "Where can I find the library?",
    "Who teaches chemistry?",
    "When is my math test?",
    "Hi there",
    "I need help with algebra",
    "How do I bake a chocolate cake?"
]


# Test prediction layer
print("\nPrediction Layer Test")
print("---------------------")

for message in messages:

    prediction, confidence = predict_intent(
        message,
        model,
        preprocessor
    )

    print(
        f"{message} -> {prediction} "
        f"(confidence: {confidence:.2f})"
    )