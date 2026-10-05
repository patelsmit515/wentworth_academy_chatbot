from src.data_loader import load_intents
from src.features import add_text_features
from src.training import train_model


# Load dataset
df = load_intents("data/intents.csv")

# Add engineered features
df = add_text_features(df)


print("\nDataset columns:")
print(df.columns.tolist())


# Train Logistic Regression baseline
(
    model,
    preprocessor,
    X_train_processed,
    X_test_processed,
    y_test,
    X_test
) = train_model(df)


print(
    f"Training rows: {X_train_processed.shape[0]}"
)

print(
    f"Number of model features: {X_train_processed.shape[1]}"
)