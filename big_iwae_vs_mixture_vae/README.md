# Big IWAE vs Mixture VAE
This directory contains the code needed to rerun the Big IWAE vs Mixture VAE experiment. The results from the experiment can be found in Table 8. 

## Mixture VAE
To recreate the Mixture VAE results in Table 8, you would run:

```
python mnist_train.py --model misvae --S S --NLayered
```
where `S = 1,2,3`.


## IWAE (increase n)

To recreate the IWAE results for an increasing number of hidden units `n` in the encoder, you would run:


```
python mnist_train.py --model misvae --L S --No_layers 1 --warmup None --h_dim n --NLayered 
```

where `n = 300,509,679` and `S = 1,2,3`. Note that when `n = 300` and `S = 1`, the kl warm up option should be changed as follows `--warmup 'kl_warmup'`




## IWAE (increase N)

To recreate the IWAE results for an increasing number of hidden layers `N` in the encoder, you would run:


```
python mnist_train.py --model misvae --device 0 --L S --warmup None --NLayered --No_layers N
```

where `N = 1,5,9` and `S = 1,2,3`. Note that when `N = 1` and `S = 1`, the kl warm up option should be changed as follows `--warmup 'kl_warmup'`


