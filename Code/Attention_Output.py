import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

tokens = ["Aero", "dynamics", "are", "for", "people", "who", "can", "'t", "build", "engine"]
input_tokens = tokens[:-1]  # 입력: 9개 토큰
target_token = tokens[-1]  # 예측 대상: "engine"

vocab = {tok: i + 1 for i, tok in enumerate(input_tokens)}
vocab_size = len(vocab) + 1

input_indices = torch.tensor([[vocab[tok] for tok in input_tokens]])


embedding_dim = 768
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
X = embedding_layer(input_indices)  # shape: (1, 9, 768)
X = X.squeeze(0)  # (9, 768)

W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = W_Q(X)  # (9, 768)
K = W_K(X)  # (9, 768)
V = W_V(X)  # (9, 768)


num_heads = 12
head_dim = embedding_dim // num_heads  # 64


def split_heads(tensor, num_heads, head_dim):
    # tensor: (seq_len, embedding_dim) -> (seq_len, num_heads, head_dim)
    seq_len = tensor.size(0)
    return tensor.view(seq_len, num_heads, head_dim)


# 각 Q, K, V를 헤드별로 분할 (9, 12, 64)
Q_heads = split_heads(Q, num_heads, head_dim)
K_heads = split_heads(K, num_heads, head_dim)
V_heads = split_heads(V, num_heads, head_dim)

# 예시로 첫 번째 헤드만 사용 (9, 64)
Q_head0 = Q_heads[:, 0, :]
K_head0 = K_heads[:, 0, :]
V_head0 = V_heads[:, 0, :]

# -----------------------------
# 4. 스케일드 점곱 어텐션 계산
# -----------------------------
# 어텐션 점수: (9, 64) x (64, 9) -> (9, 9)
scores = torch.matmul(Q_head0, K_head0.transpose(0, 1))
scaled_scores = scores / math.sqrt(head_dim)  # head_dim = 64, sqrt(64)=8

# 소프트맥스 적용: 각 토큰이 다른 토큰에 대해 주의를 기울이는 정도 (9, 9)
attn_weights = torch.softmax(scaled_scores, dim=-1)

attention_output = torch.matmul(attn_weights, V_head0)


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


plot_matrix(attention_output, "Attention Output (9 x 64)", tokens_y=input_tokens, dim_x=head_dim)
