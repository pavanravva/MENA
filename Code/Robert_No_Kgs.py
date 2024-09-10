!pip install torch_geometric
!pip install transformers

from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import DataLoader, Data
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
from transformers import RobertaModel  # Import RoBERTa model
from torch.nn import MultiheadAttention

data = torch.load('Data_for_RoBERT.pt')

data[0][3]

data_0 = []
data_1 = []
data_2 = []
for n in range(0, len(data)):
    if data[n][3] == 0:
        data_0.append(data[n])
    elif data[n][3] == 1:
        data_1.append(data[n])
    else:
        data_2.append(data[n])

for n in range(0, 4919):
    data_1.append(data_0[n])

for n in range(0, 4919):
    data_1.append(data_2[n])

len(data_1)

class GCN_RoBERTa_Combined_Model(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', gcn_hidden_dim=64, num_classes=2, num_heads=8):
        super(GCN_RoBERTa_Combined_Model, self).__init__()


        self.roberta = RobertaModel.from_pretrained(roberta_model_name)

        self.multihead_attn = MultiheadAttention(embed_dim=768, num_heads=num_heads, batch_first=True)

        self.fc1_roberta = nn.Linear(768, 128)

        self.fc_combined = nn.Linear(128, num_classes)



    def forward(self, bert_tokens, attention_mask):


        roberta_output = self.roberta(input_ids=bert_tokens, attention_mask=attention_mask)
        attn_output, _ = self.multihead_attn(roberta_output.last_hidden_state, roberta_output.last_hidden_state, roberta_output.last_hidden_state)
        roberta_representation = torch.mean(attn_output, dim=1)

        roberta_proj = F.relu(self.fc1_roberta(roberta_representation))

        return roberta_proj

gcn_roberta_model = GCN_RoBERTa_Combined_Model()



optimizer = torch.optim.Adam(gcn_roberta_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Assuming you are doing classification
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

num_epochs = 100  # Number of epochs

# Define a custom dataset for the Graph and BERT data
class GraphBERTDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        graph, asd, attention_mask, label_id = self.data_list[idx]
        if label_id in [0, 1]:
            label = 0
        else:
            label = 1
        return  graph, asd, attention_mask, label

dataset = GraphBERTDataset(data_1)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    gcn_roberta_model.train()
    for graph_data, bert_tokens, attention, label in train_loader:
        optimizer.zero_grad()

        bert_tokens = bert_tokens.squeeze(1).to(device)  # Removing extra dimension if necessary
        attention_mask = attention.squeeze(1).to(device)
        label = label.to(device)

        logits = gcn_roberta_model( bert_tokens, attention_mask)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted_label = torch.argmax(logits, dim=1)
        correct += (predicted_label == label).sum().item()
        total += label.size(0)

    train_accuracy = correct / total

    gcn_roberta_model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for graph_data, bert_tokens, attention_mask, label in val_loader:

            bert_tokens = bert_tokens.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            label = label.to(device)

            # Forward pass
            logits = gcn_roberta_model(bert_tokens, attention_mask)

            val_loss += criterion(logits, label).item()

            # Calculate accuracy on validation data
            val_correct += (torch.argmax(logits, dim=1) == label).sum().item()
            val_total += label.size(0)

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'Only_Robert_No_Kgs/{epoch+1}.pt')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

graph_data = graph_data.to(device)
            bert_tokens = bert_tokens.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            audio_data = audio_data.to(device)
            lstm_data = lstm_data.to(device)
            label = label.to(device)

            # Forward pass
            logits = model(graph_data, bert_tokens, attention_mask, audio_data, lstm_data)

            # Compute validation loss
            val_loss += criterion(logits, label).item()

            # Store predictions and labels for metric calculations
            predicted_label = torch.argmax(logits, dim=1)
            predicted_labels.extend(predicted_label.cpu().numpy())  # Store predictions
            true_labels.extend(label.cpu().numpy())  # Store true labels

            # Calculate accuracy on validation data
            val_correct += (predicted_label == label).sum().item()
            val_total += label.size(0)

val_accuracy = val_correct / val_total
avg_val_loss = val_loss / len(val_loader)

precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

scheduler.step(avg_val_loss)

print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1-Score: {f1:.4f}')
