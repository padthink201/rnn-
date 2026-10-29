"""
Recurrent Neural Network (RNN) for MNIST, written from scratch with NumPy.

Each 28x28 image is read as a sequence: 28 time steps, one row of 28 pixels per step.

Architecture (2 stacked recurrent layers):
    x_t (28)  ->  RNN layer 1 (128 units, ReLU)
              ->  RNN layer 2 (32 units, ReLU)
              ->  Dense output (10 units, softmax) on the last time step

    h1_t = relu(x_t  @ Wx1 + h1_{t-1} @ Wh1 + b1)
    h2_t = relu(h1_t @ Wx2 + h2_{t-1} @ Wh2 + b2)
    y    = softmax(h2_T @ Wy + by)

Training: cross-entropy loss, backpropagation through time (BPTT),
mini-batch gradient descent with gradient-norm clipping.
"""

import os
import time
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_mnist(path="mnist.npz"):
    """Download MNIST once (11 MB) and return normalized train/test arrays."""
    if not os.path.exists(path):
        print("Downloading MNIST...")
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as d:
        x_train, y_train = d["x_train"], d["y_train"]
        x_test, y_test = d["x_test"], d["y_test"]
    # Scale pixels to [0, 1]; shape stays (N, 28, 28) = (N, time steps, features)
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, y_train, x_test, y_test


def one_hot(y, num_classes=10):
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


# ----------------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RNN:
    def __init__(self, input_size=28, hidden1=128, hidden2=32, output_size=10, seed=0):
        rng = np.random.default_rng(seed)
        f32 = np.float32

        # Input->hidden weights: He initialization (good for ReLU)
        self.Wx1 = (rng.standard_normal((input_size, hidden1)) * np.sqrt(2.0 / input_size)).astype(f32)
        self.Wx2 = (rng.standard_normal((hidden1, hidden2)) * np.sqrt(2.0 / hidden1)).astype(f32)
        # Hidden->hidden (recurrent) weights: scaled identity ("IRNN" trick),
        # which keeps ReLU RNNs stable over many time steps
        self.Wh1 = (np.eye(hidden1) * 0.5).astype(f32)
        self.Wh2 = (np.eye(hidden2) * 0.5).astype(f32)
        self.b1 = np.zeros(hidden1, dtype=f32)
        self.b2 = np.zeros(hidden2, dtype=f32)
        # Hidden->output weights
        self.Wy = (rng.standard_normal((hidden2, output_size)) * np.sqrt(2.0 / hidden2)).astype(f32)
        self.by = np.zeros(output_size, dtype=f32)

        self.param_names = ["Wx1", "Wh1", "b1", "Wx2", "Wh2", "b2", "Wy", "by"]

    # ---------------- forward ----------------
    def forward(self, x):
        """x: (B, T, D). Returns softmax probabilities and a cache for backprop."""
        B, T, _ = x.shape
        H1, H2 = self.b1.size, self.b2.size

        # hs[t+1] is the hidden state after step t; hs[0] is the initial zero state
        h1s = [np.zeros((B, H1), dtype=np.float32)]
        h2s = [np.zeros((B, H2), dtype=np.float32)]
        z1s, z2s = [], []

        for t in range(T):
            z1 = x[:, t, :] @ self.Wx1 + h1s[-1] @ self.Wh1 + self.b1
            h1 = relu(z1)
            z2 = h1 @ self.Wx2 + h2s[-1] @ self.Wh2 + self.b2
            h2 = relu(z2)
            z1s.append(z1); h1s.append(h1)
            z2s.append(z2); h2s.append(h2)

        logits = h2s[-1] @ self.Wy + self.by
        probs = softmax(logits)
        cache = (x, z1s, h1s, z2s, h2s)
        return probs, cache

    # ---------------- backward (BPTT) ----------------
    def backward(self, probs, y_onehot, cache):
        x, z1s, h1s, z2s, h2s = cache
        B, T, _ = x.shape
        grads = {name: np.zeros_like(getattr(self, name)) for name in self.param_names}

        # Softmax + cross-entropy gradient
        dlogits = (probs - y_onehot) / B
        grads["Wy"] = h2s[-1].T @ dlogits
        grads["by"] = dlogits.sum(axis=0)

        # Gradient flowing into the last hidden state of layer 2
        dh2_next = dlogits @ self.Wy.T
        dh1_next = np.zeros_like(h1s[0])

        for t in reversed(range(T)):
            # ---- layer 2 ----
            dz2 = dh2_next * relu_grad(z2s[t])
            grads["Wx2"] += h1s[t + 1].T @ dz2
            grads["Wh2"] += h2s[t].T @ dz2
            grads["b2"] += dz2.sum(axis=0)
            dh2_next = dz2 @ self.Wh2.T          # to h2_{t-1}

            # ---- layer 1 (gets gradient from layer 2 above + from its own future) ----
            dh1 = dz2 @ self.Wx2.T + dh1_next
            dz1 = dh1 * relu_grad(z1s[t])
            grads["Wx1"] += x[:, t, :].T @ dz1
            grads["Wh1"] += h1s[t].T @ dz1
            grads["b1"] += dz1.sum(axis=0)
            dh1_next = dz1 @ self.Wh1.T          # to h1_{t-1}

        return grads

    # ---------------- gradient descent step ----------------
    def step(self, grads, lr, clip_norm=5.0):
        # Clip the global gradient norm to avoid exploding gradients in the RNN
        total_norm = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        scale = min(1.0, clip_norm / (total_norm + 1e-8))
        for name in self.param_names:
            param = getattr(self, name)
            param -= lr * scale * grads[name]   # W <- W - lr * dL/dW

    def save(self, path="rnn_weights.npz"):
        np.savez(path, **{name: getattr(self, name) for name in self.param_names})

    @classmethod
    def load(cls, path="rnn_weights.npz"):
        model = cls()
        with np.load(path) as d:
            for name in model.param_names:
                setattr(model, name, d[name])
        return model

    def predict(self, x, batch_size=1000):
        preds = []
        for i in range(0, len(x), batch_size):
            probs, _ = self.forward(x[i:i + batch_size])
            preds.append(probs.argmax(axis=1))
        return np.concatenate(preds)


def cross_entropy(probs, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(epochs=5, batch_size=64, lr=0.05, seed=0):
    x_train, y_train, x_test, y_test = load_mnist()
    y_train_oh = one_hot(y_train)

    model = RNN(seed=seed)
    rng = np.random.default_rng(seed)
    n = len(x_train)

    for epoch in range(1, epochs + 1):
        start = time.time()
        perm = rng.permutation(n)
        total_loss, total_correct = 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train_oh[idx]

            probs, cache = model.forward(xb)               # forward pass
            loss = cross_entropy(probs, yb)                # loss
            grads = model.backward(probs, yb, cache)       # backprop through time
            model.step(grads, lr)                          # gradient descent update

            total_loss += loss * len(idx)
            total_correct += np.sum(probs.argmax(axis=1) == y_train[idx])

        test_acc = np.mean(model.predict(x_test) == y_test)
        print(f"Epoch {epoch}/{epochs} | loss {total_loss / n:.4f} | "
              f"train acc {total_correct / n:.4f} | test acc {test_acc:.4f} | "
              f"{time.time() - start:.1f}s")

    model.save("rnn_weights.npz")
    print("Saved trained weights to rnn_weights.npz")
    return model


if __name__ == "__main__":
    train(epochs=5, batch_size=64, lr=0.05)
