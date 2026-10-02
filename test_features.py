from src.data_loader import load_intents
from src.features import add_text_features


df = load_intents("data/intents.csv")

df = add_text_features(df)

print("\nDataset with features:")
print(df.head())

print("\nColumns:")
print(df.columns)