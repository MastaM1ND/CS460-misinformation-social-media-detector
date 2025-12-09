"""
inference script to run DeBERTa-based misinformation classifier on Twitter posts

Usage:
   py inference_deberta_twitter.py `
       --model_dir ./deberta_twitter_out `
       --csv Datasets/ScrapedData/twitter_political_posts.csv `
       --output predictions.csv
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from tqdm import tqdm
import os

#dataset to infer
class TextDataset(Dataset):
    """
    Dataset for text data for inference.
    """
    def __init__(self, texts, tokenizer, max_len=256):
        """
        Initialize the TextDataset.
        
        :param self: Pointer to the instance of the class
        :param texts: List of text strings to be tokenized
        :param tokenizer: Tokenizer instance from Hugging Face transformers
        :param max_len: Maximum length for tokenization
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        length of the dataset.
        
        :param self: POinter to the instance of the class
        :return: Length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a tokenized item from the dataset.
        
        :param self: Pointer to the instance of the class
        :param idx: Index of the item to retrieve
        :return: Dictionary containing input_ids and attention_mask tensors
        :rtype: dict[str, torch.Tensor]
        """
        txt = self.texts[idx]
        enc = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

#Class to define the model
class TransformerClassifier(torch.nn.Module):
    """
    Transformer-based classifier model.
    """
    def __init__(self, backbone_name, num_labels=3):
        """
        Initialize the TransformerClassifier.
                
        :param self: Pointer to the instance of the class
        :param backbone_name: Name of the pretrained model backbone
        :param num_labels: Number of output labels for classification
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden // 4, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        :param self: Pointer to the instance of the class
        :param input_ids: Input tensor of token ids
        :param attention_mask: Attention mask tensor
        returns: Logits tensor for classification
        :rtype: torch.Tensor
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.classifier(pooled)

#main inference function, parse arguments and run inference
def main():
    """
    Main function to run inference using the TransformerClassifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load csv file to infer
    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print("Failed to decode utf-8-sig, falling back to latin-1.")
        df = pd.read_csv(args.csv, encoding="latin-1")

    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip().str.lower()
    
    print(f"Columns found in CSV: {df.columns.tolist()}")

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    
    def is_mostly_english(text, threshold=0.80):
        """
        Check if the text is mostly English characters.
        
        :param text: Text string to check
        :return: True if mostly English, False otherwise
        """
        if not isinstance(text, str) or len(text) == 0:
            return False
        ascii_count = sum(1 for char in text if ord(char) < 128)
        
        #keep only if above threshold
        ascii_ratio = ascii_count / len(text)
        return ascii_ratio >= threshold
    
    initial_count = len(df)

    df = df[df['text'].apply(is_mostly_english)].copy()

    #filter out non-english text in dataframe
    filtered_count = len(df)
    print(f"Filtered out {initial_count - filtered_count} entries containing too many non-ASCII characters.")
    print(f"{filtered_count} entries remaining for inference.")

    #load the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    #load the model
    model = TransformerClassifier(backbone_name=args.model_dir, num_labels=3)
    state_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Could not find trained model at {state_path}")

    torch.serialization.add_safe_globals([DebertaV2Config])

    checkpoint = torch.load(state_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    #prepare dataset and dataloader
    dataset = TextDataset(df["text"].tolist(), tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    softmax = torch.nn.Softmax(dim=1)

    #inference, loop through each batch
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(logits).cpu().numpy()
            preds = probs.argmax(axis=1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    #save predictions to csv
    df["prediction"] = all_preds
    df["conf_false"] = [p[0] for p in all_probs]
    df["conf_misinformed"] = [p[1] for p in all_probs]
    df["conf_true"] = [p[2] for p in all_probs]

    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    """
    Entry point for the script.
    """
    main()