import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from SelfAttention import MultiHeadAttentionScratch

tokens = ["Aero", "dynamics", "are", "for", "people", "who", "can", "\'t", "build", "engine"]
input_tokens = tokens[:-1]    # ["Aero", "dynamics", "are", "for", "people", "who", "can", "'t", "build"]
target_token = tokens[-1]     # "engine"

vocab = {token: idx+1 for idx, token in enumerate(tokens)}
#{'Aero': 1, 'dynamics': 2, 'are': 3, 'for': 4, 'people': 5, 'who': 6, 'can': 7, "'t": 8, 'build': 9, 'engine': 10}
vocab_size = max(vocab.values()) + 1

embedding_dim = 100 #d_model이라고도 함. 각 단어를 10 * 10 차원의 벡터로 나타냄.
num_heads = 1

input_indices = torch.tensor([[vocab[token] for token in input_tokens]]) # 결과 : tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]])
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # 각 단어를 embedding으로 수정; 11인 이유는 정답 engine까지 포함.
embeddings = embedding(input_indices) # 각 단어들을 리스트로 생성.
print(f"embedding : {embedding}\n embeddings: {embeddings}")

MultiheadAttention = MultiHeadAttentionScratch(embedding_dim, num_heads)
attn_output, attn_weights = MultiheadAttention(embeddings)

seq_len = len(input_tokens)
for head in range(num_heads):
    plt.figure(figsize=(6, 4))
    heatmap = attn_weights[0, head].detach().numpy()
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.xticks(ticks=np.arange(seq_len), labels=input_tokens, rotation=45)
    plt.yticks(ticks=np.arange(seq_len), labels=input_tokens)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title(f"Attention Head {head+1}")
    plt.tight_layout()
    plt.show()
