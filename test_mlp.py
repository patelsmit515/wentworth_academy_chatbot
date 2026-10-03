from src.data_loader import load_intents
from src.features import add_text_features
from src.training_mlp import train_mlp

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
df = load_intents("data/intents.csv")

# Add features
df = add_text_features(df)

# Train MLP
(
    model,
    preprocessor,
    X_train_processed,
    X_test_processed,
    y_test,
    X_test
) = train_mlp(df)


# Make predictions
y_pred = model.predict(X_test_processed)

# Evaluate
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMLP Neural Network Results")
print("--------------------------")

print(
    f"Accuracy: {accuracy:.2f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
    
    