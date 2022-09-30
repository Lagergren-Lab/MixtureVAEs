import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import CIFAR10
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


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

def _unnormalize(x):
    """Used to normalize discrete rgb values into the continuous range [0,1]"""
    return x*255


def load_CIFAR10(batch_size_tr=128, batch_size_val=128, batch_size_test=128, split_seed=42,  n_data_train = 100,n_data_val= 100  ):

    torch.manual_seed(split_seed)
    img_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    train_dataset = CIFAR10(root='./data/CIFAR10', download=True, train=True, transform=img_transform_train)
    test_dataset = CIFAR10(root='./data/CIFAR10', download=True, train=False, transform=img_transform)

    np.random.seed(777)

    size = len(train_dataset)
    dataset_indices = list(range(size))
    val_index = int(np.floor(0.8 * size))
    train_idx, val_idx = dataset_indices[:val_index], dataset_indices[val_index:]
    test_idx = list(range(len(test_dataset)))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_tr, shuffle=False, sampler = train_sampler)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=False, sampler = val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, sampler = test_sampler)

    return train_dataloader, val_dataloader, test_dataloader



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
