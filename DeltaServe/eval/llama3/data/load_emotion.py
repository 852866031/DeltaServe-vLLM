from datasets import load_dataset
import pandas as pd

def download_emotion_as_csv(x=100, split="train", out_path="emotion_first_x.csv"):
    ds = load_dataset("emotion", split=split)

    # Convert to pandas
    df = ds.to_pandas()

    # Keep only text + label
    df = df[["text", "label"]]

    # Map numeric → string label
    label_names = ds.features["label"].names
    df["label"] = df["label"].apply(lambda i: label_names[i])

    # Take first x rows
    df = df.head(x)

    # Save without header
    df.to_csv(out_path, index=False, header=False)
    print(f"Saved {x} samples to {out_path} (no column names).")

# Example
x = 1000
out_path = f"emotion_{x}.csv"
download_emotion_as_csv(x=x, out_path=out_path)