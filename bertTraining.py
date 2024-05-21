# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:23:58 2024

"""
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import os


# Function to load model and tokenizer
def load_model_and_tokenizer(model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Recommendation function
def recommend_car(user_input, model, tokenizer):
    input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt")
    input_ids = pad_sequence([torch.tensor(input_ids[0])], batch_first=True, padding_value=0)
    input_ids = input_ids.to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)

    predicted_label_id = torch.argmax(outputs.logits[0]).item()
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Car Name'].tolist())
    recommended_car = label_encoder.inverse_transform([predicted_label_id])[0]
    return recommended_car


# Define a Dataset class
class CommentDataset(Dataset):
    def __init__(self, comments, car_names, max_length):
        self.comments = [torch.tensor(comment[:max_length]) for comment in comments]
        self.label_encoder = LabelEncoder()
        self.car_labels = self.label_encoder.fit_transform(car_names)
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.comments[idx]),
            'labels': torch.tensor(self.car_labels[idx], dtype=torch.long)
        }
    

# Function to evaluate the model
def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy


# Read Excel file
df = pd.read_excel("data.xlsx")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Tokenize "comments" column using BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    comments = df['cleaned_text'].tolist()
    tokenized_comments = [tokenizer.encode(comment, add_special_tokens=True) for comment in comments]
    max_length = max(len(comment) for comment in tokenized_comments)
    padded_comments = pad_sequence([torch.tensor(comment) for comment in tokenized_comments], batch_first=True, padding_value=0)
    
    # Prepare data loaders
    dataset = CommentDataset(padded_comments, df['Car Name'].tolist(), max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Fine-tune BERT for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Car Name'].unique()))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Fine-tuning loop
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            print(f"Batch {i + 1}")
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    
        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)
        print(f"Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Save the model and tokenizer
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Load model and tokenizer for recommendation
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    # Example usage
    user_text = input("Enter your comment: ")
    recommended_car = recommend_car(user_text, model, tokenizer)
    print("Recommended Car:", recommended_car)