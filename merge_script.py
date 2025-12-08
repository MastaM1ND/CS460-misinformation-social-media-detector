"""
Script to make my life easier
Merges Twitter15 and Twitter16 datasets, normalizes labels in 0, 1, 2 format, and saves to a CSV.
"""

import pandas as pd

def normalize_label(label: str) -> int:
    """
    Normalize various label formats to 0 (false), 1 (unverified), 2 (true).

    :param label: Original label string
    :type label: str
    :return: Normalized label as an integer
    :rtype: int
    """
    label = label.lower().strip()
    
    mapping = {
        "true": 2,
        "real": 2,
        "false": 0,
        "fake": 0,
        "unverified": 1,
        "non-rumor": 2,
        "rumor": 1,
    }
    
    if label not in mapping:
        raise ValueError(f"Unknown label: {label}")
    
    return mapping[label]

def load_text_data(source_path, label_path):
    """
    Load tweets and their corresponding labels from given files.
    
    :param source_path: Path to the file containing source tweets
    :param label_path: Path to the file containing labels
    """
    #load tweets
    tweets = {}
    with open(source_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                tid, text = parts
                tweets[tid] = text

    #load labels
    labels = {}
    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            lbl, tid = line.split(":", 1)

            lbl = lbl.lower().strip()
            tid = tid.strip()
            numeric_label = normalize_label(lbl)
            labels[tid] = numeric_label

    #merge data
    rows = []
    for tid, text in tweets.items():
        if tid in labels:
            rows.append([tid, text, labels[tid]])

    return rows


#load our datasets
tw15 = load_text_data(
    "Datasets/twitter15/source_tweets.txt",
    "Datasets/twitter15/label.txt"
)

tw16 = load_text_data(
    "Datasets/twitter16/source_tweets.txt",
    "Datasets/twitter16/label.txt"
)

#combine the datasets
all_rows = tw15 + tw16

df = pd.DataFrame(all_rows, columns=["id", "text", "label"])

print(df.head())
print(df["label"].value_counts())

#save combined dataset
df.to_csv("Datasets/twitter15_16_merged.csv", index=False)
print("Saved → Datasets/twitter15_16_merged.csv")
