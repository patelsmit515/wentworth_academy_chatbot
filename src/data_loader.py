import pandas as pd


def load_intents(file_path):
    df = pd.read_csv(file_path)

    print(f"Number of examples: {len(df)}")

    print("\nIntent distribution:")
    print(df["intent"].value_counts())

    return df


def remove_unknown(df):
    return df[df["intent"] != "unknown"].copy()