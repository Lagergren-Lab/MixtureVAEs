import os
import datetime
import numpy as np
import torch
from data.load_data import load_mnist, load_fashion_mnist
from models.vaes import VampVAE, BetaVAE
from models.misvae import MISVAE, MISVAEwVamp, MISVAECNN, MISVAECNNwVamp
from models.HiMISVAE import HiMISVAE, HiMISVAEwVamp
from models.nfmisvae import NFMISVAE
from models.composite import Composite
import argparse


def trainer(vae, train_dataloader, val_dataloader, dir_, n_epochs=200,
            verbose=True, L=50, warmup=None, N=100, val_obj_f="miselbo", convs=False):
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
            x = x.to(vae.device).float().view((-1, 1, 28, 28))
            if not convs:
                x = x.view((-1, vae.x_dims))
            x = torch.bernoulli(x)

            loss = vae.backpropagate(x)

            train_loss_avg[epoch] += loss.item()
            num_batches += 1

        test_nll = evaluate(vae, val_dataloader, L=L, obj_f=val_obj_f, convs=convs)
        train_loss_avg[epoch] /= num_batches
        eval_loss_avg.append(test_nll)
        if test_nll < best_nll:
            path = os.path.join(dir_, "best_model")
            torch.save(vae.state_dict(), path)
            best_nll = test_nll
            best_epoch = epoch
        elif (epoch - best_epoch) >= 100:
            return train_loss_avg, eval_loss_avg

        if verbose and epoch % 10 == 0:
            print("Epoch: ", epoch)
            print(f"Test NLL: ", test_nll, f" ({round(best_nll, 2)}; {best_epoch})")
            if warmup == "kl_warmup":
                print("Beta: ", round(vae.beta, 2))

    return train_loss_avg, eval_loss_avg


def evaluate(vae, dataloader, L, obj_f='iwelbo', convs=False):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(vae.device).float().view((-1, 1, 28, 28))
        if not convs:
            x = x.view((-1, vae.x_dims))
        with torch.no_grad():
            outputs = vae(x, L)
            log_w, log_p, log_q = vae.get_log_w(x, *outputs)
            loss = vae.loss(log_w, log_p, log_q, L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def evaluate_in_parts(vae, dataloader, L, obj_f, parts=100, convs=False):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0
    if parts > L:
        print(f"parts {parts} > L {L}")
        return
    if convs:
        parts = L

    for x, y in dataloader:
        x = x.to(vae.device).float().view((-1, 1, 28, 28))
        if not convs:
            x = x.view((-1, vae.x_dims))
        with torch.no_grad():
            log_p = []
            log_q = []
            for r in range(parts):
                outputs = vae(x, L//parts)
                _, log_p_r, log_q_r = vae.get_log_w(x, *outputs)
                log_p.append(log_p_r)
                log_q.append(log_q_r)
            loss = vae.loss(_, torch.cat(log_p), torch.cat(log_q), L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def main(args):
    L_final = args.L_final
    n_epochs = 4000
    batch_size_tr = args.batch_size
    N = 100
    seed = args.seed
    K = 500
    obj_f = 'miselbo'
    device = f"cuda:{args.device}"

    if args.dataset == 'mnist':
        train_dataloader, val_dataloader, test_dataloader = load_mnist(batch_size_tr=batch_size_tr,
                                                                       batch_size_val=batch_size_tr,
                                                                       batch_size_test=100)
    elif args.dataset == 'fashion_mnist':
        train_dataloader, val_dataloader, test_dataloader = load_fashion_mnist(batch_size_tr=batch_size_tr,
                                                                               batch_size_val=batch_size_tr,
                                                                               batch_size_test=100)
    lr = args.lr
    store_path = "saved_models/mnist_models"
    warmup = args.warmup
    convs = False
    if args.model == 'misvae':
        vae = MISVAE(S=args.S, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
    elif args.model == 'misvae_vampprior':
        vae = MISVAEwVamp(S=args.S, K=K, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
    elif args.model == 'himisvae':
        vae = HiMISVAE(S=args.S, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
        convs = True
    elif args.model == 'himisvae_vampprior':
        vae = HiMISVAEwVamp(S=args.S, K=K, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
        convs = True
    elif args.model == 'nfmisvae':
        vae = NFMISVAE(S=args.S, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims, T=2)
        convs = True
    elif args.model == 'misvaecnn':
        vae = MISVAECNN(S=args.S, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
        convs = True
    elif args.model == 'misvaecnn_vampprior':
        vae = MISVAECNNwVamp(S=args.S, K=K, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims)
        convs = True
    elif args.model == 'composite':
        vae = Composite(S=args.S, K=K, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims, T=2)
        convs = True

    print("Num. params: ", count_parameters(vae))

    vae.model_name += f"_lr_{lr}_bs_{batch_size_tr}_warmup_{warmup}_N_{N}"
    folder = str(datetime.datetime.now())[0:16] + "_" + vae.model_name + f"_epochs_{n_epochs}_L_{L_final}"
    dir_ = os.path.join(store_path, folder)
    os.makedirs(dir_)
    train_loss, eval_loss = trainer(
        vae, train_dataloader, val_dataloader, dir_, n_epochs=n_epochs, L=1, warmup=warmup, N=N,
        val_obj_f=obj_f, convs=convs)
    np.save(f'{dir_}/train_loss.npy', train_loss)
    np.save(f'{dir_}/eval_loss.npy', eval_loss)
    np.save(f'{dir_}/args.npy', args)
    print("\nLoading best model\n")
    vae.load_state_dict(torch.load(os.path.join(dir_, "best_model")))
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=args.L_final, obj_f=obj_f, convs=True)
    print("Final ELBO: ", avg_elbo)
    np.save(f'{dir_}/test_elbo.npy', avg_elbo)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MISVAE')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--S', type=int, default=1)
    parser.add_argument('--model', type=str, default='misvae')
    parser.add_argument('--latent_dims', type=int, default=40)
    parser.add_argument('--warmup', type=str, default='kl_warmup')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--L_final', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()

    print(args)
    main(args)

    # for model in ["composite", "misvaecnn"]:
    #     args.model = model
    #     for S in range(1, 5):
    #         print(args)
    #         args.S = S
    #         main(args)
    """
    # vae = MISVAEwVamp(S=3, K=500, beta=1)
    # vae = MISVAE(S=1)
    # vae = HiMISVAE(S=args.S)
    vae = MISVAECNN(S=args.S)
    convs = True
    # vae = VampVAE(K=500)
    vae.load_state_dict(torch.load(os.path.join("/home/oskar/phd/misvae/saved_models/mnist_models/"
    "2022-09-15 10:28_MISVAEwCNN_a_1.0_seed_0_S_1_lr_0.0005_bs_100_warmup_kl_warmup_N_100_epochs_4000_L_5000",
                                                "best_model")))
    train_dataloader, val_dataloader, test_dataloader = load_fashion_mnist(batch_size_tr=100,
                                                                   batch_size_val=64, 
                                                                   batch_size_test=2000)
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=1000, obj_f="miselbo", convs=True)
    # avg_elbo = evaluate(vae, test_dataloader, L=5000, obj_f="miselbo")
    print("Final ELBO: ", avg_elbo)'
    """










