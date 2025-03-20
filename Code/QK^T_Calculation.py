import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math


tokens = ["Aero", "dynamics", "are", "for", "people", "who", "can", "'t", "build", "engine"]
input_tokens = tokens[:-1]
target_token = tokens[-1]

vocab = {tok: i + 1 for i, tok in enumerate(input_tokens)}
vocab_size = len(vocab) + 1

input_indices = torch.tensor([[vocab[tok] for tok in input_tokens]])

embedding_dim = 768
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
X = embedding_layer(input_indices)
X = X.squeeze(0)

W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = W_Q(X)
K = W_K(X)

num_heads = 12
head_dim = embedding_dim // num_heads  # 64

def split_heads(tensor, num_heads, head_dim):
    # tensor: (seq_len, embedding_dim) -> (seq_len, num_heads, head_dim)
    seq_len = tensor.size(0)
    return tensor.view(seq_len, num_heads, head_dim)

Q_heads = split_heads(Q, num_heads, head_dim)
K_heads = split_heads(K, num_heads, head_dim)

Q_head0 = Q_heads[:, 0, :]  # (9, 64)
K_head0 = K_heads[:, 0, :]  # (9, 64)

scores = torch.matmul(Q_head0, K_head0.transpose(0, 1))  # (9, 9)
scaled_scores = scores / math.sqrt(head_dim)  # 스케일링 (옵션), head_dim = 64 => sqrt(64)=8

def plot_heatmap(matrix, title, tokens_y=None):
    plt.figure(figsize=(6, 5))
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().numpy()
    plt.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar()
    seq_len = matrix.shape[0]
    if tokens_y is not None:
        plt.yticks(ticks=np.arange(seq_len), labels=tokens_y)
        plt.xticks(ticks=np.arange(seq_len), labels=tokens_y, rotation=45)
    else:
        plt.yticks(ticks=np.arange(seq_len))
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.show()

plot_heatmap(scaled_scores, "Scaled QK^T (9 x 9)", tokens_y=input_tokens)
