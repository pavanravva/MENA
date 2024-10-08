# -*- coding: utf-8 -*-
"""knowledge_graph.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gvthfry2UD9IF9cEkjmSNqXe9CMn807w
"""

!pip install fasttext
!pip install gensim networkx matplotlib
!pip install torch-geometric
!pip install transformers
!pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

import requests
import networkx as nx
import matplotlib.pyplot as plt
import gzip
import numpy as np
import torch
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import fasttext

class GraphProcessor:
    def __init__(self, numberbatch_path, fasttext_model):
        self.numberbatch_embeddings = self.load_numberbatch_embeddings(numberbatch_path)
        self.fasttext_model = fasttext_model
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_numberbatch_embeddings(self, file_path):
        embeddings = {}
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def get_embedding(self, word):
        if word in self.numberbatch_embeddings:
            return self.numberbatch_embeddings[word]
        else:
            fasttext_embedding = self.fasttext_model.get_word_vector(word)
            if np.count_nonzero(fasttext_embedding) == 0:
                print(f"Word '{word}' not found or has a zero vector in both Numberbatch and FastText")
                return None
            return fasttext_embedding

    def clean_token(self, token):
        return token.lstrip('Ġ')

    def tokenize_and_remove_stopwords(self, sentence):
        tokens = tokenizer.tokenize(sentence)
        filtered_tokens = [self.lemmatizer.lemmatize(self.clean_token(token.lower())) for token in tokens if self.clean_token(token.lower()) not in self.stop_words]
        return filtered_tokens


    def create_word_list(self, tokens):
        word_list = {}

        for token in tokens:
            token = token.lstrip('ġ')
            url = f'http://api.conceptnet.io/c/en/{token}'
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                else:
                    print(f"Failed to retrieve data for token '{token}', status code: {response.status_code}")
                    continue
            except requests.exceptions.RequestException as e:
                print(f"Request failed for token '{token}': {e}")
                continue
            except ValueError as e:
                print(f"JSON decode failed for token '{token}': {e}")
                continue

            english_edges = []
            for edge in data.get('edges', []):
                if 'language' in edge['start'] and 'language' in edge['end']:
                    if edge['start']['language'] == 'en' and edge['end']['language'] == 'en':
                        english_edges.append(edge)

            num_english_edges = len(english_edges)
            print(f"Number of English edges for token '{token}': {num_english_edges}")

            first_list = [edge['start']['label'].lower() for edge in english_edges]
            unique_elements = np.unique(first_list)
            if token in unique_elements:
                unique_elements = unique_elements[unique_elements != token]
            word_list[token] = unique_elements

        return word_list

    def create_and_convert_graph(self, keyword, words):
        G = nx.Graph()

        keyword_embedding = self.get_embedding(keyword)
        if keyword_embedding is not None:
            G.add_node(keyword, embedding=keyword_embedding)
        else:
            print(f"Keyword '{keyword}' not added to the graph due to missing embedding")

        for word in words:
            word_embedding = self.get_embedding(word)
            if word_embedding is not None:
                G.add_node(word, embedding=word_embedding)
                G.add_edge(keyword, word)
            else:
                print(f"Word '{word}' not added to the graph due to missing embedding")

        return G

    def combine_graphs(self, graph_list, tokens):
        combined_graph = nx.Graph()

        for G in graph_list:
            for node, data in G.nodes(data=True):
                if 'embedding' in data:
                    if node not in combined_graph:
                        combined_graph.add_node(node, embedding=data['embedding'])
                    else:
                        existing_embedding = combined_graph.nodes[node]['embedding']
                        combined_embedding = (existing_embedding + data['embedding']) / 2
                        combined_graph.nodes[node]['embedding'] = combined_embedding

            combined_graph.add_edges_from(G.edges(data=True))

        for i in range(len(tokens) - 1):
            if tokens[i] in combined_graph and tokens[i + 1] in combined_graph:
                combined_graph.add_edge(tokens[i], tokens[i + 1])

        return combined_graph

    def process_sentence(self, sentence, label):
        input_ids = []
        attention_masks = []

        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=120,
            padding='max_length',  # Use padding and truncation as per RoBERTa requirements
            truncation=True,  # Ensure the sentence is truncated if it's longer than max_length
            return_attention_mask=True,
            return_tensors='pt',
        )
        print(encoded['input_ids'].shape)
        print(encoded['attention_mask'].shape)

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        data_bert = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

        input_ids, attention_masks = data_bert

        tokens = self.tokenize_and_remove_stopwords(sentence)
        if not tokens:  # Check if tokenization failed or returned an empty list
            return None, None, None, None

        word_list = self.create_word_list(tokens)
        if not word_list:  # Check if word list creation failed or returned an empty dictionary
            return None, None, None, None

        graph_list = [self.create_and_convert_graph(key, words) for key, words in word_list.items()]
        combined_graph = self.combine_graphs(graph_list, tokens)

        if combined_graph.number_of_nodes() == 0:  # Check if the graph is empty
            return None, None, None, None

        for node in combined_graph.nodes:
            if 'embedding' not in combined_graph.nodes[node]:
                combined_graph.nodes[node]['embedding'] = np.zeros(300)

        data = from_networkx(combined_graph)
        embeddings = [combined_graph.nodes[node]['embedding'] for node in combined_graph.nodes]
        data.x = torch.tensor(embeddings, dtype=torch.float)

        return data, input_ids, attention_masks, label


# Initialize the FastText model
fasttext_model = fasttext.load_model('Embedding/cc.en.300.bin')

# Initialize the GraphProcessor with Numberbatch and FastText embeddings
numberbatch_path = 'Embedding/numberbatch-en-19.08.txt'
processor = GraphProcessor(numberbatch_path, fasttext_model)

# Load and process the CMU-MOSEI dataset
data_CMU = pd.read_csv('Final_Data_Set/CMU-MOSEI.csv')
data_CMU = data_CMU.dropna()

dataset = []
for n in range(0, len(data_CMU.iloc[:1])):
    sentence = data_CMU.iloc[n]['ASR']
    label = data_CMU.iloc[n]['Sentiment']
    print(label)
    if label == 'Negative':
        label = 0
    elif label == 'Neutral':
        label = 1
    else:
        label = 2

    processed_data = processor.process_sentence(sentence, label)
    if processed_data[0] is not None:  # Ensure the returned data is valid
        dataset.append(processed_data)
    else:
        continue

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

graph_data = a
G = to_networkx(graph_data, node_attrs=["x"], edge_attrs=None)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')

nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')

nx.draw_networkx_labels(G, pos, labels={i: i for i in range(graph_data.num_nodes)}, font_size=10)

# Display the graph
plt.title("Graph Visualization")
plt.show()

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

G = to_networkx(graph_data, node_attrs=["x"], edge_attrs=None)

node_embeddings = graph_data.x.numpy()

embedding_dim = 0
node_colors = node_embeddings[:, embedding_dim]

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap=plt.cm.viridis)

nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')

nx.draw_networkx_labels(G, pos, labels={i: i for i in range(graph_data.num_nodes)}, font_size=10)

plt.title(f"Graph Visualization (Embedding Dimension {embedding_dim})")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label=f"Embedding Value (Dimension {embedding_dim})")
plt.show()

for i, embedding in enumerate(node_embeddings):
    print(f"Node {i}: Embedding {embedding}")
