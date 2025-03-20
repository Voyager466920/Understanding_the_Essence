import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


tokens = ["Aero", "dynamics", "are", "for", "people", "who", "can", "'t", "build", "engine"]
input_tokens = tokens[:-1]  # ["Aero", "dynamics", "are", "for", "people", "who", "can", "'t", "build"]
target_token = tokens[-1]  # "engine"

# vocab 생성: 입력 토큰만 사용
vocab = {tok: i + 1 for i, tok in enumerate(input_tokens)}
vocab_size = len(vocab) + 1

# 토큰 -> 인덱스
input_indices = torch.tensor([[vocab[tok] for tok in input_tokens]])

embedding_dim = 768
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
X = embedding_layer(input_indices)  # shape: (1, 9, 768)
X = X.squeeze(0)  # (9, 768)

# Q, K, V 선형 레이어 (각각 768 -> 768)
W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = W_Q(X)  # (9, 768)
K = W_K(X)  # (9, 768)
V = W_V(X)  # (9, 768)

num_heads = 12
head_dim = embedding_dim // num_heads  # 64


# reshape: (9, 768) -> (9, 12, 64)
def split_heads(tensor, num_heads, head_dim):
    # tensor: (seq_len, embedding_dim)
    seq_len = tensor.size(0)
    return tensor.view(seq_len, num_heads, head_dim)


Q_heads = split_heads(Q, num_heads, head_dim)  # (9, 12, 64)
K_heads = split_heads(K, num_heads, head_dim)  # (9, 12, 64)
V_heads = split_heads(V, num_heads, head_dim)  # (9, 12, 64)

# 예시로 첫 번째 헤드만 선택하여 (9, 64)로 시각화
Q_head0 = Q_heads[:, 0, :]  # (9, 64)
K_head0 = K_heads[:, 0, :]  # (9, 64)
V_head0 = V_heads[:, 0, :]  # (9, 64)


def plot_matrix(matrix, title, tokens_y=None, dim_x=64):
    plt.figure(figsize=(12, 4))
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().numpy()
    plt.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar()

    seq_len = matrix.shape[0]
    if tokens_y is not None:
        plt.yticks(ticks=np.arange(seq_len), labels=tokens_y)
    else:
        plt.yticks(ticks=np.arange(seq_len))

    plt.xticks([0, dim_x - 1], [0, dim_x - 1])
    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.show()

plot_matrix(Q_head0, "Q Head 0 (9 x 64)", tokens_y=input_tokens, dim_x=head_dim)
plot_matrix(K_head0, "K Head 0 (9 x 64)", tokens_y=input_tokens, dim_x=head_dim)
plot_matrix(V_head0, "V Head 0 (9 x 64)", tokens_y=input_tokens, dim_x=head_dim)
