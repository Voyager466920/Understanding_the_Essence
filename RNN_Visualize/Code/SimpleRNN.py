import numpy as np
from Utility_Functions import softmax, one_hot

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, learning_rate=1e-1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.U = np.random.randn(hidden_size, vocab_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(vocab_size, hidden_size) * 0.01

    def forward(self, x, h_prev):
        self.h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h_prev))
        self.y = np.dot(self.V, self.h)
        self.p = softmax(self.y)
        return self.h, self.p

    def backward(self, x, h_prev, target):
        dy = np.copy(self.p)
        dy[target] -= 1

        dV = np.outer(dy, self.h)  # (vocab_size x hidden_size)
        dh = np.dot(self.V.T, dy)  # (hidden_size,)
        # tanh의 미분: 1 - h^2
        dtanh = (1 - self.h ** 2) * dh  # (hidden_size,)
        dU = np.outer(dtanh, x)  # (hidden_size x vocab_size)
        dW = np.outer(dtanh, h_prev)  # (hidden_size x hidden_size)

        # 파라미터 업데이트
        self.V -= self.learning_rate * dV
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW

        dh_prev = np.dot(self.W.T, dtanh)
        return dh_prev

    def predict(self, seed_word, h_prev, word_to_idx, idx_to_word):
        x = one_hot(word_to_idx[seed_word], self.vocab_size)
        h, p = self.forward(x, h_prev)
        next_ix = np.argmax(p)
        return idx_to_word[next_ix], h
