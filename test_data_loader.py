from src.data_loader import load_intents


df = load_intents("data/intents.csv")

print("\nFirst five rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())