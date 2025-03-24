import numpy as np
import matplotlib.pyplot as plt

from Understanding_the_Essence.RNN_Visualize.Code.SimpleRNN import SimpleRNN
from Understanding_the_Essence.RNN_Visualize.Code.Utility_Functions import one_hot

# -------------------------------
# 1. 데이터 및 사전 생성
# -------------------------------
corpus = ("Aerodynamics are for people who can not build engine. "
          "Those who study aerodynamics discover a world where the subtle interplay of forces and fluid dynamics sparks innovative ideas, "
          "inspiring them to harness the beauty of airflow and design efficient solutions even when traditional engineering skills fall short.").lower().split()

# 단어 사전을 알파벳 순으로 정렬하여 생성 (시각화할 때 순서가 일정하도록)
vocab = sorted(set(corpus))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# 입력, 타겟 시퀀스 생성
inputs  = [word_to_idx[word] for word in corpus[:-1]]
targets = [word_to_idx[word] for word in corpus[1:]]

# -------------------------------
# 2. RNN 모델 초기화 및 학습
# -------------------------------
hidden_size = 10
learning_rate = 0.1
rnn = SimpleRNN(vocab_size, hidden_size, learning_rate)

num_epochs = 50  # 간소화를 위해 에포크 수를 줄임
for epoch in range(num_epochs):
    total_loss = 0
    h_prev = np.zeros(hidden_size)
    for x_ix, t_ix in zip(inputs, targets):
        x = one_hot(x_ix, vocab_size)
        h_prev, p = rnn.forward(x, h_prev)
        total_loss += -np.log(p[t_ix] + 1e-8)
        h_prev = rnn.backward(x, h_prev, t_ix)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}")

# -------------------------------
# 3. 한 문장에 대한 순전파 결과 수집
# -------------------------------
T = len(corpus)  # 시간 스텝 수
x_all = []
h_all = []
o_all = []
h_prev = np.zeros(hidden_size)

for t in range(T):
    x_t = one_hot(word_to_idx[corpus[t]], vocab_size)
    x_all.append(x_t)
    h_prev, o_t = rnn.forward(x_t, h_prev)
    h_all.append(h_prev)
    o_all.append(o_t)

# 배열로 변환
x_matrix = np.array(x_all)  # shape: (T, vocab_size)
h_matrix = np.array(h_all)  # shape: (T, hidden_size)
o_matrix = np.array(o_all)  # shape: (T, vocab_size)

# -------------------------------
# 4. 히트맵 시각화 함수 (축 라벨 지원)
# -------------------------------
def plot_heatmap(matrix, title, xlabel="Columns", ylabel="Rows", xticklabels=None, yticklabels=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), xticklabels, rotation=90)
    if yticklabels is not None:
        plt.yticks(range(len(yticklabels)), yticklabels)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. 히트맵 시각화
# -------------------------------
# (a) 입력 벡터 x (타임스텝 x vocab), x축: 단어 라벨, y축: 시간 스텝 (문장에서 단어 순서)
plot_heatmap(x_matrix,
             title="Input X (One-hot) Over Time",
             xlabel="Vocab", ylabel="Time Step",
             xticklabels=[idx_to_word[i] for i in range(vocab_size)])

# (b) 은닉 상태 h (타임스텝 x hidden_size), y축: 시간 스텝, x축: 은닉 유닛 인덱스
plot_heatmap(h_matrix,
             title="Hidden States H Over Time",
             xlabel="Hidden Unit", ylabel="Time Step")

# (c) 출력 o (타임스텝 x vocab), x축: 단어 라벨, y축: 시간 스텝
plot_heatmap(o_matrix,
             title="Output O (Logits) Over Time",
             xlabel="Vocab", ylabel="Time Step",
             xticklabels=[idx_to_word[i] for i in range(vocab_size)])

# (d) 가중치 행렬 U (Input-to-Hidden): shape (hidden_size, vocab_size)
plot_heatmap(rnn.U,
             title="Weight Matrix U (Input-to-Hidden)",
             xlabel="Vocab", ylabel="Hidden Unit",
             xticklabels=[idx_to_word[i] for i in range(vocab_size)])

# (e) 가중치 행렬 W (Hidden-to-Hidden): shape (hidden_size, hidden_size)
plot_heatmap(rnn.W,
             title="Weight Matrix W (Hidden-to-Hidden)",
             xlabel="Hidden Unit", ylabel="Hidden Unit")

# (f) 가중치 행렬 V (Hidden-to-Output): shape (vocab_size, hidden_size)
plot_heatmap(rnn.V,
             title="Weight Matrix V (Hidden-to-Output)",
             xlabel="Hidden Unit", ylabel="Vocab",
             yticklabels=[idx_to_word[i] for i in range(vocab_size)])
