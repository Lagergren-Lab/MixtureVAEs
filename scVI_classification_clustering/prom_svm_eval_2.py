import torch
from prom_lock_misvae import MISVAE as lock_MISVAE
from prom_misvae import MISVAE

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import anndata


### TRUE LABELS ###
gene_dataset = anndata.read_h5ad('cortex_anndata')  #'PL_ver1'

cell_types = np.array ( gene_dataset.obs["cell_type"], dtype = str)
true_labels = np.zeros_like (cell_types)
true_labels[np.where(cell_types == 'astrocytes_ependymal')] = 1
true_labels[np.where(cell_types == 'endothelial-mural')] = 2
true_labels[np.where(cell_types == 'interneurons')] = 3
true_labels[np.where(cell_types == 'microglia')] = 4
true_labels[np.where(cell_types == 'oligodendrocytes')] = 5
true_labels[np.where(cell_types == 'pyramidal CA1')] = 6
true_labels[np.where(cell_types == 'pyramidal SS')] = 7

test_idx = np.load('test_idx.npy')
test_labels = true_labels[test_idx]

mydev = torch.device('cuda:2')

model1 = MISVAE (n_input= 1200, S=1, device=mydev)
model2 = MISVAE (n_input= 1200, S=2, device=mydev)
model3 = MISVAE (n_input= 1200, S=3, device=mydev)
model4 = MISVAE (n_input= 1200, S=4, device=mydev)
model5 = MISVAE (n_input= 1200, S=5, device=mydev)
model6 = lock_MISVAE (n_input= 1200, S=2, device=mydev)
model7 = lock_MISVAE (n_input= 1200, S=3, device=mydev)
model8 = lock_MISVAE (n_input= 1200, S=4, device=mydev)
model9 = lock_MISVAE (n_input= 1200, S=5, device=mydev)

model1.load_state_dict(torch.load("/home/semihkurt/prom_misvae/model1"), strict=False)
model2.load_state_dict(torch.load("/home/semihkurt/prom_misvae/model2"), strict=False)
model3.load_state_dict(torch.load("/home/semihkurt/prom_misvae/model3"), strict=False)
model4.load_state_dict(torch.load("/home/semihkurt/prom_misvae/model4"), strict=False)
model5.load_state_dict(torch.load("/home/semihkurt/prom_misvae/model5"), strict=False)
model6.load_state_dict(torch.load("/home/semihkurt/prom_misvae/lock_model2"), strict=False)
model7.load_state_dict(torch.load("/home/semihkurt/prom_misvae/lock_model3"), strict=False)
model8.load_state_dict(torch.load("/home/semihkurt/prom_misvae/lock_model4"), strict=False)
model9.load_state_dict(torch.load("/home/semihkurt/prom_misvae/lock_model5"), strict=False)


model_list = [model1,model2,model3,model4,model5,model6,model7,model8,model9]

score= np.zeros([200,14])
# 2-4-6-8-10
clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-4, C=0.025))

indices = np.random.permutation(405)
svm_train_idx, svm_test_idx = indices[:305], indices[305:]

for rep_exp in range(200):

    for j in range(9):
        if j == 0:
            for km in range (5):
                z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=(1+km)*2)
                z_p = z.permute(1, 0, 2, 3)
                z_r = torch.reshape(z_p, [405, 10*(1+km)*2])
                z_r = z_r.cpu().detach().numpy()

                svm_train, svm_test = z_r[svm_train_idx, :], z_r[svm_test_idx, :]

                clf.fit(svm_train, test_labels[svm_train_idx])
                score[rep_exp, km] = clf.score(svm_test, test_labels[svm_test_idx])

                #clf.fit(z_r, test_labels)
                #score[km] = clf.score(z_r, test_labels)

            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](
                torch.tensor(gene_dataset.X[test_idx, :]), L=1)
            qz_m = torch.reshape(qz_m, [405, 10 * (j + 1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            svm_train, svm_test = latent_para[svm_train_idx, :], latent_para[svm_test_idx, :]

            clf.fit(svm_train, test_labels[svm_train_idx])
            score[rep_exp, j + 5] = clf.score(svm_test, test_labels[svm_test_idx])

        elif j < 5:
            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=1)
            qz_m = torch.reshape(qz_m,[405,10*(j+1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            svm_train, svm_test = latent_para[svm_train_idx, :], latent_para[svm_test_idx, :]

            clf.fit(svm_train, test_labels[svm_train_idx])
            score[rep_exp, j+5] = clf.score(svm_test, test_labels[svm_test_idx])

            #clf.fit(latent_para, test_labels)
            #score[j + 4] = clf.score(latent_para, test_labels)

        else:
            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=1)
            qz_m = torch.reshape(qz_m, [405, 10 * (j - 4 + 1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j - 4 + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            svm_train, svm_test = latent_para[svm_train_idx, :], latent_para[svm_test_idx, :]

            clf.fit(svm_train, test_labels[svm_train_idx])
            score[rep_exp, j + 5] = clf.score(svm_test, test_labels[svm_test_idx])

score = score.mean(axis=0)
print('score vanilla vaes:')
print(score[:5])
print('score misvae-fixed decoder:')
print(score[10:])
print('score misvae:')
print(score[5:10])






































