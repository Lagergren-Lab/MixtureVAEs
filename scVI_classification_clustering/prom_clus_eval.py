import torch
from prom_misvae import MISVAE
from prom_lock_misvae import MISVAE as lock_MISVAE
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import anndata

mydev = 'cuda:2'

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
num_clus = np.unique(test_labels).shape[0]
print('number of clusters:')
print(num_clus)

asw = np.zeros([200, 14])
ari = np.zeros([200, 14])
nmi = np.zeros([200, 14])
model_list = [model1,model2,model3,model4,model5,model6,model7,model8,model9]


for rep_exp in range(200):

    for j in range(9):
        if j == 0:
            for km in range (5):
                z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=(1+km)*2)
                z_p = z.permute(1, 0, 2, 3)
                z_r = torch.reshape(z_p, [405, 10*(1+km)*2])
                z_r = z_r.cpu().detach().numpy()

                ### K-MEANS ###
                kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(z_r)
                pred_labels = kmeans.labels_

                asw[rep_exp, km] = metrics.silhouette_score(z_r, pred_labels, metric='euclidean')
                ari[rep_exp, km] = metrics.adjusted_rand_score(test_labels, pred_labels)
                nmi[rep_exp, km] = metrics.normalized_mutual_info_score(test_labels, pred_labels)

            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](
                torch.tensor(gene_dataset.X[test_idx, :]), L=1)
            qz_m = torch.reshape(qz_m, [405, 10 * (j + 1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            ### K-MEANS ###
            kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(latent_para)
            pred_labels = kmeans.labels_

            asw[rep_exp, j + 5] = metrics.silhouette_score(latent_para, pred_labels, metric='euclidean')
            ari[rep_exp, j + 5] = metrics.adjusted_rand_score(test_labels, pred_labels)
            nmi[rep_exp, j + 5] = metrics.normalized_mutual_info_score(test_labels, pred_labels)

        elif j < 5:
            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=1)
            qz_m = torch.reshape(qz_m,[405,10*(j+1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            ### K-MEANS ###
            kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(latent_para)
            pred_labels = kmeans.labels_

            asw[rep_exp, j+5] = metrics.silhouette_score(latent_para, pred_labels, metric='euclidean')
            ari[rep_exp, j+5] = metrics.adjusted_rand_score(test_labels, pred_labels)
            nmi[rep_exp, j+5] = metrics.normalized_mutual_info_score(test_labels, pred_labels)


        else:
            z, qz_m, qz_v, library, ql_m, ql_v, px_rate, px_r, px_dropout = model_list[j](torch.tensor(gene_dataset.X[test_idx,:]), L=1)
            qz_m = torch.reshape(qz_m, [405, 10 * (j - 4 + 1)])
            qz_v = torch.reshape(qz_v, [405, 10 * (j - 4 + 1)])
            latent_para = torch.concat((qz_m, qz_v), 1)
            latent_para = latent_para.cpu().detach().numpy()

            ### K-MEANS ###
            kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(latent_para)
            pred_labels = kmeans.labels_

            asw[rep_exp, j+5] = metrics.silhouette_score(latent_para, pred_labels, metric='euclidean')
            ari[rep_exp, j+5] = metrics.adjusted_rand_score(test_labels, pred_labels)
            nmi[rep_exp, j+5] = metrics.normalized_mutual_info_score(test_labels, pred_labels)



asw = asw.mean(axis=0)
ari = ari.mean(axis=0)
nmi = nmi.mean(axis=0)

print('asw vanilla vaes:')
print(asw[:5])
print('asw misvae-fixed decoder:')
print(asw[10:])
print('asw misvae:')
print(asw[5:10])

print('ari vanilla vaes:')
print(ari[:5])
print('ari misvae-fixed decoder:')
print(ari[10:])
print('ari misvae:')
print(ari[5:10])

print('nmi vanilla vaes:')
print(nmi[:5])
print('nmi misvae-fixed decoder:')
print(nmi[10:])
print('nmi misvae:')
print(nmi[5:10])
