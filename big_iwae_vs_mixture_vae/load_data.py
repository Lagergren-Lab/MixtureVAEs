import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(batch_size_tr=128, batch_size_val=128, batch_size_test=128, split_seed=42):
    torch.manual_seed(split_seed)
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)

    # seed for binomial
    np.random.seed(777)

    # train_samples = np.random.binomial(1, train_dataset.data[:50000] / 255)
    train_samples = train_dataset.data[:50000] / 255
    train_labels = train_dataset.targets[:50000]

    train = TensorDataset(train_samples, train_labels)
    train_dataloader = DataLoader(train, batch_size=batch_size_tr, shuffle=True)

    val_labels = train_dataset.targets[50000:]
    val_samples = np.random.binomial(1, train_dataset.data[50000:] / 255)
    val = TensorDataset(torch.from_numpy(val_samples), val_labels)
    val_dataloader = DataLoader(val, batch_size=batch_size_val, shuffle=True)

    test_samples = np.random.binomial(1, test_dataset.data / 255)
    test_labels = test_dataset.targets
    test = TensorDataset(torch.from_numpy(test_samples), test_labels)
    test_dataloader = DataLoader(test, batch_size=batch_size_test, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_bimodal(seed=42, n=5000, x_dims=2, mean=3, bs=124, pi_=0.5):
    np.random.seed(seed)
    n_clusters = 2
    z_dims = 2
    Z = []
    pi = np.array([0.8, .2])

    for i in range(n_clusters):
        sigma = np.eye(z_dims)

        mu = np.zeros(z_dims) + mean * (-1) ** i
        z = np.random.multivariate_normal(mu, sigma, size=int(n * pi[i]))
        Z.append(z)
    Z = np.concatenate(Z, 0)

    mu = np.sign(Z) * mean
    X = np.random.normal(mu, np.ones_like(mu))

    n_train = 4900
    train_dataloader = DataLoader(X.reshape(n, x_dims), batch_size=bs, shuffle=True)
    return train_dataloader, X, Z


def load_shell(dr=0.1, R=1):
    z = np.random.uniform(-2, 2, size=(10000, 2))
    r = np.linalg.norm(z, 2, axis=-1)
    # p = ((r >= R) & (r <= R + dr)).astype(int)
    p = np.exp(-(r - R + dr) ** 2 / .01)
    return z, p


if __name__ == '__main__':
    z, p = load_shell()
    plt.scatter(z[:, 0], z[:, 1], c=p)
    plt.colorbar()
    plt.show()
