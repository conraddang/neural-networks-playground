import pandas as pd
import numpy as np
from torchvision import datasets

def download_data():
    train_data = datasets.MNIST(root='data', train=True, download=True)
    test_data = datasets.MNIST(root='data', train=False, download=True)
    print("MNIST dataset downloaded...")
    return train_data, test_data

def prepare_data(train_data, test_data):
    X_train = train_data.data.numpy().reshape(train_data.data.shape[0], -1).T / 255.0
    y_train = train_data.targets.numpy()
    X_test = test_data.data.numpy().reshape(test_data.data.shape[0], -1).T / 255.0
    y_test = test_data.targets.numpy()
    print("\nData prepared...")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def init_model_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    print("\nModel parameters initialized...")
    print(f"W1: {W1.shape}, b1: {b1.shape}, W2: {W2.shape}, b2: {b2.shape}")
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    return Z2

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def loss_fn(logits, y):
    pass

def backprop():
    pass

def gradient_step():
    pass

def main():
    train_data, test_data = download_data()
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)
    W1, b1, W2, b2 = init_model_params()
    logits = forward(X_train[:, 0].reshape(-1, 1), W1, b1, W2, b2)
    print(logits)
    probs = softmax(logits)
    print(probs)
    pred = np.argmax(logits, axis=0)
    print(pred)
    loss = loss_fn(logits, y_train[0])

if __name__ == "__main__":
    main()