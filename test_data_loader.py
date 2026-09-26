from src.data_loader import load_intents

df = load_intents("data/intents.csv")

print(df.head())