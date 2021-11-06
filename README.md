# CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator
-----
This is the official code repository for NeurIPS 2021 paper: [CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator](https://arxiv.org/pdf/2110.14002.pdf)
by [Alek Dimitriev](https://alekdimi.github.io/) and [Mingyuan Zhou](https://mingyuanzhou.github.io/).

To install the required packages run: *pip install -r requirements.txt*
To reproduce the toy example run: *python3 toy.py*.
Supported gradients: CARMS, LOORF, UNORD, ARSM.
Supported datasets: Dynamic MNIST, Fashion MNIST, and Omniglot, with either a linear or nonlinear encoder/decoder pair. 
To run an experiment you can use the following template:
```
python3 -m main \
    --dataset=dynamic_mnist \
    --logdir=../logs \
    --ckptdir=../ckpts \
    --grad_type=carms \
    --num_samples=4 \
    --num_latent=10 \
    --num_categories=5 \
    --encoder_type=nonlinear \
    --repeat_idx=42 \
    --num_steps=1e6 \
    --demean_input \
    --initialize_with_bias \
```
