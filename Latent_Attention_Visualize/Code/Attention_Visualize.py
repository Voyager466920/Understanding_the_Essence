import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Understanding_the_Essence.Latent_Attention_Visualize.Code.Latent_Attetion import LatentAttention


plt.style.use('dark_background')


embed_dim = 768
latent_dim = 576
att_module = LatentAttention(embed_dim, latent_dim)

sentence = "aerodynamics is for people who can't build engines."
tokens = sentence.split()

X = torch.rand(1, len(tokens), embed_dim)


Q = att_module.proj_q(X)     # Q = X(W_QW_{UK}^T)
L_kv = att_module.proj_kv(X)   # L_{KV} = X(W_{DKV})

attn_scores = torch.matmul(Q, L_kv.transpose(-2, -1)) / att_module.scale
attn_weights = F.softmax(attn_scores, dim=-1)

# X
plt.figure(figsize=(10, 6))
plt.imshow(X[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel("Embedding Dimension")
plt.ylabel("Tokens")
plt.yticks(range(len(tokens)), tokens)
plt.title("Input Embeddings X")
plt.show()

# Softmax(Q * L_kv^T)
plt.figure(figsize=(8, 6))
plt.imshow(attn_weights[0].detach().cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.title("Attention Heatmap (Softmax(Q*L_kv^T))")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.show()

# W_Q heatmap
plt.figure(figsize=(10, 6))
plt.imshow(att_module.proj_q.weight.detach().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel("Embedding Dimension")
plt.ylabel("Latent Dimension")
plt.title("W_Q")
plt.show()

# W_KV heatmap
plt.figure(figsize=(10, 6))
plt.imshow(att_module.proj_kv.weight.detach().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel("Embedding Dimension")
plt.ylabel("Latent Dimension")
plt.title("W_KV")
plt.show()

# X(W_QW_{UK}^T)
plt.figure(figsize=(10, 6))
plt.imshow(Q[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Latent Dimension")
plt.ylabel("Tokens")
plt.title("Q = X(W_QW_{UK}^T)")
plt.show()

# L_{KV} = X(W_{DKV})
plt.figure(figsize=(10, 6))
plt.imshow(L_kv[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Latent Dimension")
plt.ylabel("Tokens")
plt.title("L_{KV} = X(W_{DKV})")
plt.show()
