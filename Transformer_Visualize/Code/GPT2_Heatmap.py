from transformers import GPT2Tokenizer, GPT2Model
import torch
import matplotlib.pyplot as plt
import numpy as np

# 사전 학습된 GPT-2 토크나이저와 모델 불러오기 (어텐션 가중치 출력 활성화)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# 분석할 문장: "Aerodynamics are for people who can't build"
text = "Aerodynamics are for people who can't build"
inputs = tokenizer(text, return_tensors="pt")

# 모델 실행: 출력에 어텐션 정보 포함
outputs = model(**inputs)
# outputs.attentions는 튜플 형태이며, 각 요소의 shape는 (batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions

# 토큰 목록: GPT-2의 토크나이저는 byte-level BPE를 사용하므로 토큰이 다소 분리되어 나올 수 있음.
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
seq_len = len(tokens)

# GPT-2의 총 레이어 수와 각 레이어의 헤드 수
num_layers = len(attentions)         # 보통 12
num_heads = attentions[0].shape[1]     # 보통 12

# 12 x 12 그리드의 서브플롯 생성 (그림 크기는 조절 가능)
fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads*2, num_layers*2))

for layer_idx in range(num_layers):
    # 해당 레이어의 어텐션 가중치: (num_heads, seq_len, seq_len)
    attn_layer = attentions[layer_idx][0].detach().numpy()
    for head in range(num_heads):
        ax = axes[layer_idx, head]
        heatmap = attn_layer[head]  # (seq_len, seq_len)
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        # 복잡한 토큰 레이블은 너무 촘촘할 수 있으므로, 가독성을 위해 가장 바깥쪽에만 토큰 라벨을 표시합니다.
        if layer_idx == num_layers - 1:
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        else:
            ax.set_xticklabels([])
        if head == 0:
            ax.set_yticklabels(tokens, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"L{layer_idx+1} H{head+1}", fontsize=6)

plt.tight_layout()
plt.show()
