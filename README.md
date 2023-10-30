# Dis-inhibitory neuronal circuits can control the sign of synaptic plasticity

This repository contains the code associated with the NeurIPS 2023 paper "Dis-inhibitory neuronal circuits can control the sign of synaptic plasticity".

## Installation

The following instructions should allow you to replicate our results in a fresh virtual environment. We used Python version 3.8.10.

#### 1. Installing JAX with GPU support

Installing JAX with GPU support is required to replicate our results. You can use the following command:

`pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Note that the exact version you need might depend on the CUDA version on your local hardware. (see https://github.com/google/jax#installation for detailed instructions)

#### 2. Other packages

All other required packages can be installed by running 
`pip install -r requirements.txt`

## Replicating Results 

#### 1. Replicating figures

To replicate the experimental results we report in Figures 3 & 4 and Supplementary Figures S1 & S2, refer to the respective jupyter notebooks inside the `notebooks` folder.

#### 2. Training multi-layer networks on computer vision benchmarks

For training of multi-layer networks on computer vision benchmarks, first go to the `conf/dataset` directory and change the path variable in `fmnist.yaml` and `mnist.yaml` to a local directory of your choice. 

We use `hydra` for experiment logging and configuration. Simulations are run by calling the `main.py` file. The following commands train some of the networks we report in the manuscript:

- To train a 3-Layer MLP with classic backprop on Fashion-MNIST:
```
python main.py dataset=fmnist model/vf=non-dale trainer=bp model.vf.nb_hidden=3
```

- To train a 3-Layer MLP with dis-inhibitory control using the "exact inverse" learning rule:
```
python main.py dataset=fmnist model/vf=dale trainer=exact-inv model.vf.nb_hidden=3
```

- To train a 3-Layer MLP with dis-inhibitory control using the "linear threshold" learning rule:
```
python main.py dataset=fmnist model/vf=dale trainer=lin-thresh model.vf.nb_hidden=3
```

To train networks on MNIST instead of Fashion-MNIST, simply change `fmnist` to `mnist` in the commands above. To use the average Jacobian as feedback weights, add `trainer.average_fb_weights=True` to the commands.

All other simulation parameters can be be overridden using the same syntax as above or changed in the `conf` directory (refer to the https://hydra.cc/ documentation for details on the configuration files).




