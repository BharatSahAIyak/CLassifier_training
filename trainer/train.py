
import pandas as pd
import numpy as np
from datasets import load_dataset



import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
#from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

df = pd.read_csv('data/train.csv')

df['sentence'] = df['sentence'].str.replace('?', '', regex=False)
df_duplicated = df.copy()

# Step 2: Add a question mark at the end of all sentences in the duplicated DataFrame
df_duplicated['sentence'] = df_duplicated['sentence'] + '?'

# Step 3: Concatenate the original DataFrame with the modified duplicate
df = pd.concat([df, df_duplicated], ignore_index=True)


# Map the labels
df['label'] = df['class'].map(label_mapping)

df = df.reset_index(drop = True )

model_name = 'google-bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=df['class'].nunique())

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

max_length = 128
train_encodings = tokenizer(list(train_df['sentence']), truncation=True, padding=True, max_length=max_length)

val_encodings = tokenizer(list(val_df['sentence']), truncation=True, padding=True, max_length=max_length)

# Convert the tokenized data to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_df['label'].values))

val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_df['label'].values))

# Create data loaders for training and validation
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Evaluation

def model_eval  ():
  model.eval()
  all_predicted_labels = []
  all_true_labels = []

  with torch.no_grad():
      for batch in val_dataloader:
          inputs = {'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'labels': batch[2].to(device)}
          outputs = model(**inputs)
          logits = outputs.logits
          predicted_labels = torch.argmax(logits, dim=1)
          all_predicted_labels.extend(predicted_labels.cpu().numpy())
          all_true_labels.extend(inputs['labels'].cpu().numpy())

  # Calculate accuracy and F1 score
  val_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
  val_f1_score = f1_score(all_true_labels, all_predicted_labels, average='weighted')

  print(f"Validation Accuracy: {val_accuracy:.4f}")
  print(f"Validation F1 Score: {val_f1_score:.4f}")
  return(val_accuracy,val_f1_score)

# Set the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)
from sklearn.metrics import accuracy_score, f1_score

# Training loop
epochs = 1000
model.to(device)
model.train()

prev_loss = float('inf')
no_decrease_count = 0

for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predicted_labels = []
    all_true_labels = []

    for batch in train_dataloader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted_labels = torch.argmax(outputs.logits, dim=1)
        true_labels = inputs['labels']
        total_loss += loss.item()
        total_correct += (predicted_labels == true_labels).sum().item()
        total_samples += len(true_labels)
        all_predicted_labels.extend(predicted_labels.tolist())
        all_true_labels.extend(true_labels.tolist())

    # Calculate accuracy and F1 score
    epoch_loss = total_loss / len(train_dataloader)
    epoch_accuracy = total_correct / total_samples
    epoch_f1_score = f1_score(all_true_labels, all_predicted_labels, average='weighted')

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f} - F1 Score: {epoch_f1_score:.4f}")

    # Check if the loss decreased
    if epoch_loss >= prev_loss:
        no_decrease_count += 1
    else:
        no_decrease_count = 0

    prev_loss = epoch_loss

    model_eval()

    # Stop training if there is no decrease in loss for 2 consecutive iterations
    if no_decrease_count >= 5:
        print("No decrease in loss for 2 consecutive iterations. Stopping training.")
        break

model_eval()

model.save_pretrained('output_model/.')
tokenizer.save_pretrained('output_model/.')
