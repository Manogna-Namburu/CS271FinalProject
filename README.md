# CS271FinalProject: VAE Image Blurriness Experiments
This project studies the causes of blurry image reconstructions in Variational Autoencoders (VAEs). The main research question is:

**What factors contribute most to VAE image blurriness, and which modifications are most effective in reducing it?**

This project uses the CIFAR-10 dataset from TorchVision.

## Experiment:
The following VAE configurations are tested:

| Experiment | Description |
|---|---|
| Baseline | MSE loss, latent dimension 20, hidden dimension 400, beta = 1.0 |
| BCE Loss | Changes reconstruction loss from MSE to BCE |
| Beta 0.1 | Reduces KL regularization strength |
| Beta 5.0 | Increases KL regularization strength |
| Latent 10 | Reduces latent dimension to 10 |
| Latent 50 | Increases latent dimension to 50 |
| Big Decoder | Increases hidden dimension from 400 to 800 |


## To Run The Experiment Script From Terminal:
python3 VAE_experiments.py

It will download CIFAR-10, train the baseline VAE, train each modified VAE model, save reconstruction images in the results folder, and print loss values, blur scores, inference times, and robustness blur scores
