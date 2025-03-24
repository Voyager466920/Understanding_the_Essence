import numpy as np
from Understanding_the_Essence.RNN_Visualize.Code.SimpleRNN import SimpleRNN
from Understanding_the_Essence.RNN_Visualize.Code.Utility_Functions import one_hot

# 코퍼스 준비
corpus = ("Aerodynamics are for people who can not build engine. "
          "Those who study aerodynamics discover a world where the subtle interplay of forces and fluid dynamics sparks innovative ideas, "
          "inspiring them to harness the beauty of airflow and design efficient solutions even when traditional engineering skills fall short."
         ).lower().split()

# 단어 사전 생성
vocab = sorted(set(corpus))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# 입력/타겟 시퀀스
inputs  = [word_to_idx[w] for w in corpus[:-1]]
targets = [word_to_idx[w] for w in corpus[1:]]

# RNN 초기화
hidden_size = 50
learning_rate = 0.1
rnn = SimpleRNN(vocab_size, hidden_size, learning_rate)

# 학습 (Epoch 수를 줄여 간소화)
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    h_prev = np.zeros(hidden_size)
    for x_ix, t_ix in zip(inputs, targets):
        x = one_hot(x_ix, vocab_size)
        h_prev, p = rnn.forward(x, h_prev)
        total_loss += -np.log(p[t_ix] + 1e-8)
        h_prev = rnn.backward(x, h_prev, t_ix)

# 최종 손실 출력
print(f"Final Loss: {total_loss:.4f}")

# 시드 단어부터 몇 개 단어를 연속 예측
seed_word = "are"
h_prev = np.zeros(hidden_size)
generated = [seed_word]
num_generate = 5  # 생성할 단어 개수

for _ in range(num_generate):
    x = one_hot(word_to_idx[generated[-1]], vocab_size)
    h_prev, p = rnn.forward(x, h_prev)
    next_idx = np.argmax(p)
    generated.append(idx_to_word[next_idx])

print("Generated Sequence:")
print(" ".join(generated))
