import os
import sys

conf_path = os.getcwd()
sys.path.append(conf_path)

from prom_misvae import MISVAE
from prom_lock_misvae import MISVAE as lock_MISVAE

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import numpy as np
#import scanpy as sc
import anndata
#import pandas as pd
#import argparse

from torch.utils.data import Dataset, DataLoader, random_split, Subset

dseed = 42

gene_dataset = anndata.read_h5ad('cortex_anndata')

print('heyo')
mydev = torch.device('cuda:2')
#print(gene_dataset.X.shape)

test_idx = np.load('test_idx.npy')
val_idx = np.load('val_idx.npy')
training_idx = np.load('training_idx.npy')

training_data, val_data, test_data = gene_dataset.X[training_idx, :], gene_dataset.X[val_idx, :], gene_dataset.X[test_idx, :]

train_loader = DataLoader(training_data, batch_size=512)
val_loader = DataLoader(val_data, batch_size=200)
test_loader = DataLoader(test_data, batch_size=200)

for data in train_loader:
    data = data.to(mydev)

for data in val_loader:
    data = data.to(mydev)

for data in test_loader:
    data = data.to(mydev)

model1 = MISVAE (n_input= 1200, S=1, device=mydev) #device=cuda
model1.load_state_dict(torch.load("model1"), strict=False)

with torch.cuda.device(2):
    model6 = lock_MISVAE (n_input= 1200, S=2, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model2")
    model6.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model7 = lock_MISVAE (n_input= 1200, S=3, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model3")
    model7.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model8 = lock_MISVAE (n_input= 1200, S=4, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model4")
    model8.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model9 = lock_MISVAE (n_input= 1200, S=5, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model5")
    model9.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)


    torch.save(model6.state_dict(), "/home/semihkurt/svm_models_2/lock_model2")
    torch.save(model7.state_dict(), "/home/semihkurt/svm_models_2/lock_model3")
    torch.save(model8.state_dict(), "/home/semihkurt/svm_models_2/lock_model4")
    torch.save(model9.state_dict(), "/home/semihkurt/svm_models_2/lock_model5")










