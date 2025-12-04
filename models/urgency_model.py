# File: train_urgency_model.py (Corrected and Edited for Class Imbalance)

from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from safetensors.torch import save_file
import pandas as pd
from tqdm import tqdm
import numpy as np # <-- CHANGE 1: Added numpy import

# --- Configuration ---
class Config:
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data/attack_data.csv")
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "urgency_model.safetensors")
    TOKENIZER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "bert_tokenizer/") 
    LABEL_ENCODER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "urgency_encoder.pkl")
    
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_dataset(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=config.MAX_LEN, return_tensors="pt")
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels, dtype=torch.long))

def train_epoch(model, data_loader, loss_fn, optimizer):
    model.train()
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

def evaluate_model(model, data_loader, label_encoder):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    
    # (This is your good fix, which I am keeping)
    all_possible_labels = range(len(label_encoder.classes_))
    
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=label_encoder.classes_,
        labels=all_possible_labels, # This is the fix
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}\nClassification Report:\n{report}")

def save_artifacts(model, tokenizer):
    print("\nSaving model...")
    save_file(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Urgency model saved to {config.MODEL_SAVE_PATH}")

def main():
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Dataset file not found at: {config.DATA_PATH}")
    data = pd.read_csv(config.DATA_PATH)
    
    if 'urgency' not in data.columns or data['urgency'].isnull().any():
        raise ValueError("Dataset is missing the 'urgency' column or it contains empty values. Please add it to 'attack_data_cleaned.csv'.")

    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data["urgency"])
    num_labels = len(label_encoder.classes_)
    
    with open(config.LABEL_ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Urgency label encoder with {num_labels} classes saved.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["description"], data['label_encoded'], test_size=0.2, random_state=42, stratify=data['label_encoded']
    )

    # --- CHANGE 2: Calculate Class Weights for Imbalanced Data ---
    print("Calculating class weights for imbalance...")
    
    # We use the y_train (numerical labels) to calculate weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),  # Get unique encoded labels (e.g., [0, 1, 2])
        y=y_train                    # Pass all training labels
    )
    
    # Convert weights to a PyTorch tensor and send to the active device
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Using weights for classes {np.unique(y_train)}: {weights_tensor}")
    # --- End of Class Weight Calculation ---


    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH) 
    model = BertForSequenceClassification.from_pretrained(config.PRE_TRAINED_MODEL_NAME, num_labels=num_labels).to(device)

    train_dataset = create_dataset(tokenizer, X_train.tolist(), y_train.tolist())
    test_dataset = create_dataset(tokenizer, X_test.tolist(), y_test.tolist())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- CHANGE 3: Pass the weights tensor to the loss function ---
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        train_epoch(model, train_loader, loss_fn, optimizer)

    print("\n--- Evaluating model performance on test set ---")
    evaluate_model(model, test_loader, label_encoder)
    save_artifacts(model, tokenizer)

if __name__ == "__main__":
    main()