"""
train_deberta_twitter.py

Training script for DeBERTa-based misinformation classifier on Twitter 15 and 16 datasets
The script loads the merged dataset CSV, fine-tunes a DeBERTa model, and saves the best model.
In classifying, the script converts Twitter posts into three classes: false (0), unverified (1), true (2).

Usage:
    py train_deberta_twitter.py `
        --csv Datasets/twitter15_16_merged.csv `
        --out_dir ./deberta_twitter_out `
        --model_name microsoft/deberta-v3-small `
        --epochs 6 `
        --batch_size 16 `
        --unfreeze_layers 1 `
        --augment
"""

import argparse
import math
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, logging
from typing import List, Dict, Any

logging.set_verbosity_error()

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="CSV with columns: id,text,label (0=false,1=unverified,2=true)")
parser.add_argument("--out_dir", default="./deberta_twitter_out", help="Where to save model and tokenizer")
parser.add_argument("--model_name", default="microsoft/deberta-v3-small", help="Hugging Face model")
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr_head", type=float, default=2e-4, help="LR for classifier head")
parser.add_argument("--lr_encoder", type=float, default=1e-5, help="LR for unfrozen encoder layers")
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--val_size", type=float, default=0.15)
parser.add_argument("--unfreeze_layers", type=int, default=1, help="How many top encoder layers to unfreeze (0 = none)")
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on val macro-F1")
parser.add_argument("--augment", action="store_true", help="Enable simple train-time augmentation")
parser.add_argument("--device", default=None, help="cuda or cpu (auto if not given)")
args = parser.parse_args()

#set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
os.makedirs(args.out_dir, exist_ok=True)

def random_deletion(words: List[str], p=0.1) -> List[str]:
    """
    Randomly delete words from the list with probability p.
        
    :param words: List of words to potentially delete from
    :type words: List[str]
    :param p: Probability of deleting each word
    :return: List of words after random deletion
    :rtype: List[str]
    """
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p]

def random_swap(words: List[str], swap_times=1) -> List[str]:
    """
    Randomly swap two words in the list a given number of times.
    
    :param words: List of words to potentially swap
    :type words: List[str]
    :param swap_times: Number of times to swap words
    :return: List of words after random swaps
    :rtype: List[str]
    """
    words = words.copy()
    for _ in range(swap_times):
        i = random.randrange(len(words))
        j = random.randrange(len(words))
        words[i], words[j] = words[j], words[i]
    return words

def augment_text(text: str) -> str:
    """
    Apply simple augmentation to the input text.
        
    :param text: Input text to augment
    :type text: str
    :return: Augmented text
    :rtype: str
    """
    words = text.split()
    if len(words) <= 2:
        return text
    op = random.choice(["del", "swap", "none", "none"])  
    if op == "del":
        words = random_deletion(words, p=0.08)
    elif op == "swap":
        words = random_swap(words, swap_times=1)
    return " ".join(words)

class TextDataset(Dataset):
    """
    Dataset for text classification.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int, augment=False):
        """
        Initializes the TextDataset.

        :param self: Pointer to the instance of the class
        :param texts: List of input texts
        :type texts: List[str]
        :param labels: List of labels corresponding to the texts
        :type labels: List[int]
        :param tokenizer: Tokenizer to convert text to tokens
        :param max_len: Maximum length for tokenization
        :type max_len: int
        :param augment: Whether to apply augmentation
        :type augment: bool
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        """
        Returns the length of the dataset.
                
        :param self: Pointer to the instance of the class
        :return: Length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        :param self: Pointer to the instance of the class
        :param idx: Index of the item to retrieve
        :type idx: int
        :return: Dictionary containing the text and label
        :rtype: Dict[str, Any]
        """
        txt = self.texts[idx]
        if self.augment:
            txt = augment_text(txt)
        return {"text": txt, "label": self.labels[idx]}

def collate_fn(batch: List[Dict[str,Any]], tokenizer, max_len: int):
    """
    Collate function to prepare batches for DataLoader.

    :param batch: List of samples in the batch
    :type batch: List[Dict[str, Any]]
    :param tokenizer: Tokenizer to convert text to tokens
    :param max_len: Maximum length for tokenization
    :type max_len: int
    """
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"], labels

#model definition
class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier with optional layer unfreezing.
    """
    def __init__(self, backbone_name: str, num_labels: int, dropout: float = 0.3, unfreeze_last_n: int = 0):
        """
        Initializes the TransformerClassifier.
        
        :param self: Pointer to the instance of the class
        :param backbone_name: Name of the pretrained transformer backbone
        :type backbone_name: str
        :param num_labels: Number of output labels/classes
        :type num_labels: int
        :param dropout: Dropout rate for the classifier
        :type dropout: float
        :param unfreeze_last_n: Number of last encoder layers to unfreeze
        :type unfreeze_last_n: int
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.config = self.backbone.config
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_labels)
        )

        #freeze all layers initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        # unfreeze last N layers if specified
        if unfreeze_last_n > 0:
            n_layers = getattr(self.config, "num_hidden_layers", None)
            if n_layers is None:
                max_idx = -1
                for n, _ in self.backbone.named_parameters():
                    import re
                    m = re.search(r"encoder\.layer\.(\d+)\.", n)
                    if m:
                        max_idx = max(max_idx, int(m.group(1)))
                if max_idx >= 0:
                    n_layers = max_idx + 1

            if n_layers is not None:
                start_unfreeze = max(0, n_layers - unfreeze_last_n)
                #unfreeze layers
                for name, param in self.backbone.named_parameters():
                    if f"encoder.layer." in name:
                        import re
                        m = re.search(r"encoder\.layer\.(\d+)", name)
                        if m:
                            layer_idx = int(m.group(1))
                            if layer_idx >= start_unfreeze:
                                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        :param self: Pointer to the instance of the class
        :param input_ids: Input token IDs
        :param attention_mask: Attention mask for input tokens
        :return: Logits from the classifier
        :rtype: torch.Tensor
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq_out = out.last_hidden_state  # (batch, seq, hidden)
        attn = attention_mask.unsqueeze(-1)
        pooled = (seq_out * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        return logits

#training metrics
def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute evaluation metrics: macro F1, precision, recall.
    
    :param y_true: True labels
    :type y_true: List[int]
    :param y_pred: Predicted labels
    :type y_pred: List[int]
    :return: Dictionary containing macro F1, precision, recall, and F1 scores
    :rtype: Dict[str, float]
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"macro_f1": macro_f1, "precision": p, "recall": r, "f1": f1}

#loading dataset
print("Loading CSV:", args.csv)
df = pd.read_csv(args.csv)

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must contain 'text' and 'label' columns (label values: 0=false,1=unverified,2=true)")

#clean data
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

#train/val split
train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed, stratify=df["label"])

print(f"Train/Val sizes: {len(train_df)}/{len(val_df)} (labels distribution)")
print(train_df["label"].value_counts().to_dict(), val_df["label"].value_counts().to_dict())

#tokenizer, datasets, dataloaders
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len, augment=args.augment)
val_dataset = TextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_len, augment=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))

#model, optimizer, scheduler, criterion
num_labels = len(df["label"].unique())
model = TransformerClassifier(args.model_name, num_labels=num_labels, dropout=0.3, unfreeze_last_n=args.unfreeze_layers)
model.to(device)

#optimizer with layer-specific LRs
head_params = [p for n, p in model.named_parameters() if n.startswith("classifier") or "classifier" in n]
encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("classifier") or "classifier" in n)]

param_groups = []
if head_params:
    param_groups.append({"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay})
if encoder_params:
    param_groups.append({"params": encoder_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay})

if not param_groups:
    param_groups = [{"params": model.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay}]

optimizer = AdamW(param_groups)

num_training_steps = len(train_loader) * args.epochs
warmup_steps = max(1, math.ceil(0.06 * num_training_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

#Vals for training loop
best_val_f1 = -1.0
best_epoch = -1
no_improve = 0

scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
print("Starting training on device:", device)

#training loop
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
    for input_ids, attention_mask, labels in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        train_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{train_loss / (len(all_labels)+1):.4f}", "acc": f"{(np.array(all_preds)==np.array(all_labels)).mean():.4f}"})

    train_loss = train_loss / len(train_dataset)
    train_metrics = compute_metrics(all_labels, all_preds)

    #validation
    model.eval()
    val_loss = 0.0
    v_preds = []
    v_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch} val"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            v_preds.extend(preds.tolist())
            v_labels.extend(labels.detach().cpu().numpy().tolist())

    val_loss = val_loss / len(val_dataset)
    val_metrics = compute_metrics(v_labels, v_preds)
    val_f1 = val_metrics["macro_f1"]

    print(f"Epoch {epoch} summary: train_loss={train_loss:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

    #stop early to prevent overfitting
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        no_improve = 0
        save_path = os.path.join(args.out_dir, "best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "tokenizer_name_or_path": args.model_name,
            "config": getattr(model.backbone, "config", None)
        }, save_path)
        #save tokenizer
        tokenizer.save_pretrained(args.out_dir)
        print(f"New best model saved (epoch {epoch}) with val_macro_f1={val_f1:.4f} -> {save_path}")
    else:
        no_improve += 1
        print(f"No improvement for {no_improve} epoch(s) (best {best_val_f1:.4f} at epoch {best_epoch})")

    if no_improve >= args.patience:
        print(f"Early stopping after {epoch} epochs (no improvement for {args.patience} epochs).")
        break

#save model
model_to_save = model.backbone if hasattr(model, "backbone") else model
model_to_save.save_pretrained(args.out_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.save_pretrained(args.out_dir)
print(f"Training complete. Model and tokenizer saved to {args.out_dir}")