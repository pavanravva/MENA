!pip install torch_geometric
!pip install transformers

from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import os
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
from transformers import RobertaModel  # Import RoBERTa model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

data_1 = torch.load('UCF_data_Skeleton_MFCC_with_labels.pt')

lables_values = []
for n in range(0, len(data_1)):
    lables_values.append(data_1[n][5])

pd.DataFrame(lables_values).value_counts()

zeros = np.zeros(300)

list_of_zeros_embeddings  = []
for n in range(0, len(data_1)):
    a = data_1[n][0]
    embeddings = a.embedding
    shape_emd = embeddings.shape
    for m in range(0, shape_emd[0]):
        if embeddings[m] == zeros:
            list_of_zeros_embeddings.append(a)
            break

len(list_of_zeros_embeddings)

all_data = data_1
filtered_data = [item for item in all_data if item is not None]

class GCN_RoBERTa_Combined_Model(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', gcn_hidden_dim=64, num_classes=3, num_heads=8):
        super(GCN_RoBERTa_Combined_Model, self).__init__()

        # GCN layers
        self.gcn1 = GCNConv(300, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.dropout = nn.Dropout(0.5)

        # Load RoBERTa model
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)

        # Multihead attention layer
        self.multihead_attn = MultiheadAttention(embed_dim=768, num_heads=num_heads, batch_first=True)

        # Fully connected layers for each model
        self.fc1_graph = nn.Linear(gcn_hidden_dim, 128)
        self.fc1_roberta = nn.Linear(768, 128)

    def forward(self, graph_data, bert_tokens, attention_mask):
        # GCN forward pass
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)

        graph_representation = global_mean_pool(x, batch)

        roberta_output = self.roberta(input_ids=bert_tokens, attention_mask=attention_mask)
        attn_output, _ = self.multihead_attn(roberta_output.last_hidden_state, roberta_output.last_hidden_state, roberta_output.last_hidden_state)
        roberta_representation = torch.mean(attn_output, dim=1)

        graph_proj = F.relu(self.fc1_graph(graph_representation))
        roberta_proj = F.relu(self.fc1_roberta(roberta_representation))

        return graph_proj, roberta_proj


# CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Prepare for LSTM (batch_size, timesteps, features)

        # Initialize hidden and cell states with correct shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))  # Forward propagate LSTM
        return out[:, -1, :]  # Use last hidden state of LSTM

# Loading pretrained GCN-RoBERTa and CNN-LSTM models
gcn_roberta_model = GCN_RoBERTa_Combined_Model()
cnn_lstm_model = CNNLSTMModel(input_size=128, hidden_size=128, num_layers=2, num_classes=3)

gcn_roberta_weights_path = 'Robert_a_weights/final.pt'
cnn_lstm_weights_path = 'Final_Wieghts_For_Combing/CNN_100.pt'

gcn_roberta_model.load_state_dict(torch.load(gcn_roberta_weights_path), strict = False)
cnn_lstm_model.load_state_dict(torch.load(cnn_lstm_weights_path), strict = False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last time step
        return out

lstm_model = LSTMModel(input_size=99, hidden_size=64, num_layers=4, num_classes=3)

lstm_weights_path = 'Final_Wieghts_For_Combing/lstm_model_epoch_99.pth'

lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=torch.device('cpu')), strict=False)

class CombinedModelWithAttention(nn.Module):
    def __init__(self, gcn_roberta_model, cnn_lstm_model, lstm_model, combined_hidden_dim=192, num_classes=2, num_heads=8):
        super(CombinedModelWithAttention, self).__init__()
        self.gcn_roberta = gcn_roberta_model
        self.cnn_lstm = cnn_lstm_model
        self.lstm_model = lstm_model
        self.multihead_attn = MultiheadAttention(embed_dim=combined_hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc_output = nn.Linear(combined_hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc_equal_em = nn.Linear(256, 192)
        self.fc_equal_concat = nn.Linear(192, 192)

    def forward(self, graph_data, bert_tokens, attention_mask, audio_data, lstm_data):
        graph_proj, roberta_proj = self.gcn_roberta(graph_data, bert_tokens, attention_mask)
        gcn_roberta_features = torch.cat((graph_proj, roberta_proj), dim=1)

        cnn_lstm_proj = self.cnn_lstm(audio_data)
        lstm_proj = self.lstm_model(lstm_data)
        combined_qv_features = torch.cat((cnn_lstm_proj, lstm_proj), dim=1)


        gcn_roberta_features = gcn_roberta_features.unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
        combined_qv_features = combined_qv_features.unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)

        att_k = self.fc_equal_em(gcn_roberta_features)

        att_v = self.fc_equal_concat(combined_qv_features)

        attn_output, attn_weights = self.multihead_attn(query=att_v, key=att_k, value=att_v)

        attn_output = self.dropout(F.relu(attn_output.squeeze(1)))  # Remove the sequence dimension (squeeze)

        output = self.fc_output(attn_output)

        return output

model = CombinedModelWithAttention(gcn_roberta_model, cnn_lstm_model, lstm_model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Assuming you are doing classification
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

num_epochs = 100  # Number of epochs

class GraphBERTDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        graph_data, bert_tokens, attention_mask, mfcc_data, lstm_skeleton, label, label_name = self.data_list[idx]
        mfcc_data = torch.tensor(mfcc_data, dtype=torch.float32)
        return graph_data, bert_tokens, attention_mask, mfcc_data, lstm_skeleton, label, label_name

dataset = GraphBERTDataset(data_1)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GraphBERTDataset_path = 'Robert_a_weights/10_Combined_model_weights.pt'

model.load_state_dict(torch.load(GraphBERTDataset_path, map_location=torch.device('cpu')), strict=False)



for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    model.train()
    for graph_data, bert_tokens, attention_mask, audio_data, lstm_data, label, label_name in train_loader:
        optimizer.zero_grad()

        graph_data = graph_data.to(device)
        bert_tokens = bert_tokens.squeeze(1).to(device)  # Removing extra dimension if necessary
        attention_mask = attention_mask.squeeze(1).to(device)
        audio_data = audio_data.to(device)
        lstm_data = lstm_data.to(device)
        label = label.to(device)

        logits = model(graph_data, bert_tokens, attention_mask, audio_data, lstm_data)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted_label = torch.argmax(logits, dim=1)
        correct += (predicted_label == label).sum().item()
        total += label.size(0)

    train_accuracy = correct / total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for graph_data, bert_tokens, attention_mask, audio_data, lstm_data, label, lable_nam in val_loader:
            # Move tensors to the correct device
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
        torch.save(model.state_dict(), f'Robert_a_weights/{epoch+1}.pt')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

true_labels = []
predicted_labels = []

with torch.no_grad():
        for graph_data, bert_tokens, attention_mask, audio_data, lstm_data, label, lable_nam in val_loader:
            # Move tensors to the correct device
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

