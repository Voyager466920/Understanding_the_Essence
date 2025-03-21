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
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

num_heads = 12
head_dim = embedding_dim // num_heads


def split_heads(tensor, num_heads, head_dim):
    seq_len = tensor.size(0)
    return tensor.view(seq_len, num_heads, head_dim)


Q_heads = split_heads(Q, num_heads, head_dim)
K_heads = split_heads(K, num_heads, head_dim)
V_heads = split_heads(V, num_heads, head_dim)

attn_outputs = []
for i in range(num_heads):
    Q_i = Q_heads[:, i, :]
    K_i = K_heads[:, i, :]
    V_i = V_heads[:, i, :]
    scores = torch.matmul(Q_i, K_i.transpose(0, 1))
    scaled_scores = scores / math.sqrt(head_dim)
    attn_probs = torch.softmax(scaled_scores, dim=-1)
    attention_output = torch.matmul(attn_probs, V_i)
    attn_outputs.append(attention_output)

fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 3, 4), constrained_layout=True)
for i, ax in enumerate(axes):
    mat = attn_outputs[i].detach().numpy()
    im = ax.imshow(mat, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_title(f"Head {i + 1}", fontsize=8)
    ax.set_xticks([0, head_dim - 1])
    ax.set_xticklabels([0, head_dim - 1], fontsize=6)
    ax.set_yticks(np.arange(len(input_tokens)))
    ax.set_yticklabels(input_tokens, fontsize=6)

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.suptitle("Attention Output per Head (9 x 64)", fontsize=12)
plt.show()
