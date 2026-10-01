from src.features import create_text_features


messages = [
    "Who teaches chemistry?",
    "I have to go!",
    "What is my class today?"
]


features = create_text_features(messages)

print(features)