import os
import sys

conf_path = os.getcwd()
sys.path.append(conf_path)

from prom_misvae import MISVAE
#from prom_lock_misvae import MISVAE as lock_MISVAE

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import numpy as np
#import scanpy as sc
import anndata
#import pandas as pd
#import argparse

from torch.utils.data import Dataset, DataLoader, random_split, Subset

dseed = 42
#gene_dataset = _cortex._load_cortex()
#gene_dataset = scvi.data.heart_cell_atlas_subsampled()
#gene_dataset = scvi.data.cortex()
#gene_dataset.subsample_genes(1000, mode="variance")
#gene_dataset.make_gene_names_lower()
#gene_dataset.write('cortex_anndata')
#gene_dataset = anndata.read_h5ad("/Users/semih.kurt/Documents/misvae_scvi/cortex_anndata")  #'PL_ver1'
gene_dataset = anndata.read_h5ad('cortex_anndata')

print('heyo')
mydev = torch.device('cuda:2')
#print(gene_dataset.X.shape)

indices = np.random.permutation(gene_dataset.X.shape[0])
training_idx, val_idx, test_idx = indices[:2400], indices[2400:2600] , indices[2600:]

np.save('training_idx', training_idx)
np.save('val_idx', val_idx)
np.save('test_idx', test_idx)

print('indices saved!')

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

with torch.cuda.device(2):
    model1 = MISVAE (n_input= 1200, S=1, device=mydev, model_name="model1") #device=cuda
    model1.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model2 = MISVAE (n_input= 1200, S=2, device=mydev, model_name="model2")
    model2.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model3 = MISVAE (n_input= 1200, S=3, device=mydev, model_name="model3")
    model3.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model4 = MISVAE (n_input= 1200, S=4, device=mydev, model_name="model4")
    model4.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    model5 = MISVAE (n_input= 1200, S=5, device=mydev, model_name="model5")
    model5.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    #model6 = lock_MISVAE (n_input= 1200, S=2, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model2")
    #model6.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    #model7 = lock_MISVAE (n_input= 1200, S=3, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model3")
    #model7.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    #model8 = lock_MISVAE (n_input= 1200, S=4, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model4")
    #model8.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    #model9 = lock_MISVAE (n_input= 1200, S=5, device=mydev, decoder=model1.decoder, px_r=model1.px_r, model_name="lock_model5")
    #model9.trainer(n_epochs=1000, train_dataloader=train_loader, val_dataloader=val_loader)

    torch.save(model1.state_dict(), "/home/semihkurt/svm_models_2/model1")
    torch.save(model2.state_dict(), "/home/semihkurt/svm_models_2/model2")
    torch.save(model3.state_dict(), "/home/semihkurt/svm_models_2/model3")
    torch.save(model4.state_dict(), "/home/semihkurt/svm_models_2/model4")
    torch.save(model5.state_dict(), "/home/semihkurt/svm_models_2/model5")

    #torch.save(model6.state_dict(), "/home/semihkurt/svm_models_2/lock_model2")
    #torch.save(model7.state_dict(), "/home/semihkurt/svm_models_2/lock_model3")
    #torch.save(model8.state_dict(), "/home/semihkurt/svm_models_2/lock_model4")
    #torch.save(model9.state_dict(), "/home/semihkurt/svm_models_2/lock_model5")










