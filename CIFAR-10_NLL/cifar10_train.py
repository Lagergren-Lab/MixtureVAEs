import os
import datetime
import numpy as np
import torch
from load_data import load_CIFAR10
from models.misvae import MISVAE, MISVAEwVamp, DropMISVAE, MISVAEwGMM
from models.decoders import CNNDecoder, CNNDecoderLogMix

def trainer(vae, train_dataloader, val_dataloader, dir_, n_epochs=200,
            verbose=True, L=50, warmup=None, N=100, val_obj_f="miselbo"):
    if warmup == "kl_warmup":
        vae.beta = 0
    vae.train()
    train_loss_avg = np.zeros(n_epochs)
    eval_loss_avg = []
    best_nll = 1e10
    best_epoch = 0

    for epoch in range(n_epochs):
        num_batches = 0

        if warmup == "kl_warmup":
            vae.beta = np.minimum(1 / (N - 1) * epoch, 1.)

        for x, y in train_dataloader:
            x = x.to(vae.device).float()

            # forward
            recon, z, mu, std = vae.forward(x)
            # backward
            loss = vae.backpropagate(x, z, mu, std, recon)

            train_loss_avg[epoch] += loss.item()
            num_batches += 1

        test_nll = evaluate(vae, val_dataloader, L=L, obj_f=val_obj_f)
        train_loss_avg[epoch] /= num_batches
        eval_loss_avg.append(test_nll)
        if test_nll < best_nll:
            if epoch > 100:
                path = os.path.join(dir_, "best_model")
                torch.save(vae.state_dict(), path)

            best_nll = test_nll
            best_epoch = epoch
        elif (epoch - best_epoch) >= 100:
            return train_loss_avg, eval_loss_avg

        if verbose and epoch % 10 == 0:
            print("Epoch: ", epoch)
            print(f"Test NLL: ", test_nll, f" ({round(best_nll, 2)}; {best_epoch})")
            print(f"Test BPD: ", test_nll / (32 ** 2 * 3) / np.log(2), f" ({round(best_nll / (32 ** 2 * 3) / np.log(2), 2)}; {best_epoch})")
            if warmup == "kl_warmup":
                print("Beta: ", round(vae.beta, 2))

    return train_loss_avg, eval_loss_avg


def evaluate(vae, dataloader, L, obj_f='iwelbo'):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(vae.device)
        with torch.no_grad():

            recon, z, mu, std = vae.forward(x, L)
            log_w, log_p, log_q = vae.get_log_w(x, z, mu, std, recon)
            loss = vae.loss(log_w, log_p, log_q, L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def evaluate_in_parts(vae, dataloader, L, obj_f):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0
    parts = 200
    if parts > L:
        print(f"parts {parts} > L {L}")
        return


    for x, y in dataloader:
        x = x.to(vae.device).float()
        with torch.no_grad():
            log_p = []
            log_q = []
            for r in range(parts):
                recon, z, mu, std  = vae(x, L//parts)
                _, log_p_r, log_q_r = vae.get_log_w(x, z, mu, std, recon)
                log_p.append(log_p_r)
                log_q.append(log_q_r)
            loss = vae.loss(_, torch.cat(log_p), torch.cat(log_q), L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def main():
    L = 1
    L_final = 5000
    n_epochs = 15
    batch_size_tr = 100
    N = 100
    seed = 0
    K = 100
    latent_dims = 128
    S = 3
    cifar10_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    no_channels = 3

    lr = 0.0005
    obj_f = 'miselbo'


    train_dataloader, val_dataloader, test_dataloader = load_CIFAR10(batch_size_tr=batch_size_tr,
                                                                     batch_size_val=batch_size_tr,
                                                                     batch_size_test=64)

    store_path = "saved_models/cifar_models"
    warmup = "kl_warmup"


    # vae = MISVAE(S=S, beta=1., lr=lr, seed=seed, L=L, device = device, x_dims = cifar10_size, z_dims= latent_dims, no_channels = no_channels, decoder = CNNDecoder(n_dims = cifar10_size, latent_dims= latent_dims, no_channels = no_channels))
    vae = MISVAE(S=S, beta=1., lr=lr, seed=seed, L=L, device=device, x_dims=cifar10_size, z_dims=latent_dims,
                 no_channels=no_channels,
                 decoder=CNNDecoderLogMix(n_dims=cifar10_size, latent_dims=latent_dims, no_channels=no_channels))

    vae.to(device)

    print("Num. params: ", count_parameters(vae))
    print("S = ", vae.S)

    vae.model_name += f"_lr_{lr}_bs_{batch_size_tr}_warmup_{warmup}_N_{N}"
    folder = str(datetime.datetime.now())[0:20] + "_" + vae.model_name + f"_epochs_{n_epochs}_L_{L_final}"
    dir_ = os.path.join(store_path, folder)
    os.makedirs(dir_)
    train_loss, eval_loss = trainer(
        vae, train_dataloader, val_dataloader, dir_, n_epochs=n_epochs, L=L, warmup=warmup, N=N, val_obj_f=obj_f)

    np.save(f'{dir_}/train_loss.npy', train_loss)
    np.save(f'{dir_}/eval_loss.npy', eval_loss)
    
    print("\nLoading best model\n")
    vae.load_state_dict(torch.load(os.path.join(dir_, "best_model")))
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=L_final, obj_f=obj_f)
    print("Final ELBO: ", avg_elbo)

    np.save(f'{dir_}/test_elbo.npy', avg_elbo)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    main()
