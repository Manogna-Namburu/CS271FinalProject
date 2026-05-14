import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

INPUT_DIM = 3 * 32 * 32

print("Using device:", device)

# --------------------------
# DATA
# --------------------------
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --------------------------
# VAE MODEL
# --------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, INPUT_DIM)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# --------------------------
# LOSS FUNCTION
# --------------------------
def loss_function(recon_x, x, mu, logvar, beta=1.0, loss_type="mse"):
    if loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

# --------------------------
# TRAIN FUNCTION
# --------------------------
def train(model, optimizer, beta=1.0, loss_type="mse"):
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for x, _ in train_loader:
        x = x.view(-1, INPUT_DIM).to(device)

        optimizer.zero_grad()

        recon, mu, logvar = model(x)

        loss, recon_loss, kl_loss = loss_function(
            recon,
            x,
            mu,
            logvar,
            beta=beta,
            loss_type=loss_type
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_kl_loss = total_kl_loss / len(train_loader.dataset)

    return avg_loss, avg_recon_loss, avg_kl_loss

# --------------------------
# BLUR
# --------------------------
def blur_score(image):

    if image.ndim == 3:
        image = np.mean(image, axis=2)  # convert RGB to grayscale

    gx, gy = np.gradient(image)

    return np.var(gx) + np.var(gy)

# --------------------------
# SAVE RECONSTRUCTIONS
# --------------------------
def save_images(model, name):
    model.eval()

    x, _ = next(iter(train_loader))
    x = x.view(-1, INPUT_DIM).to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    x = x.view(-1, 3, 32, 32).cpu()
    recon = recon.view(-1, 3, 32, 32).cpu()

    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))

    for i in range(8):
        original_img = np.transpose(x[i].numpy(), (1, 2, 0))
        recon_img = np.transpose(recon[i].numpy(), (1, 2, 0))

        axes[0, i].imshow(original_img)
        axes[1, i].imshow(recon_img)

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"results/{name}.png")
    plt.close()

    scores = [
        blur_score(np.transpose(img.numpy(), (1, 2, 0)))
        for img in recon[:8]
    ]

    print(f"{name} Blur Score: {np.mean(scores):.6f}")

def measure_inference_time(model, name, num_batches=20):
    model.eval()
    times = []

    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= num_batches:
                break

            x = x.view(-1, INPUT_DIM).to(device)

            start = time.time()
            recon, _, _ = model(x)
            end = time.time()

            times.append(end - start)

    avg_time = np.mean(times)
    print(f"{name} Average Inference Time per Batch: {avg_time:.6f} seconds")
    print(f"{name} Average Inference Time per Image: {avg_time / BATCH_SIZE:.8f} seconds")


# --------------------------
# ROBUSTNESS TESTING
# --------------------------
def apply_corruption(x, corruption_type):
    if corruption_type == "noise":
        noise = 0.2 * torch.randn_like(x)
        x_corrupt = x + noise
        return torch.clamp(x_corrupt, 0, 1)

    elif corruption_type == "dark":
        x_corrupt = x * 0.5
        return torch.clamp(x_corrupt, 0, 1)

    elif corruption_type == "bright":
        x_corrupt = x * 1.5
        return torch.clamp(x_corrupt, 0, 1)

    else:
        return x


def robustness_test(model, name):
    model.eval()

    x, _ = next(iter(train_loader))
    x = x.to(device)

    corruptions = ["noise", "dark", "bright"]

    os.makedirs("results", exist_ok=True)

    for corruption in corruptions:
        x_corrupt = apply_corruption(x, corruption)
        x_flat = x_corrupt.view(-1, INPUT_DIM)

        with torch.no_grad():
            recon, _, _ = model(x_flat)

        original = x.cpu()
        corrupted = x_corrupt.cpu()
        recon = recon.view(-1, 3, 32, 32).cpu()

        fig, axes = plt.subplots(3, 8, figsize=(12, 5))

        for i in range(8):
            axes[0, i].imshow(np.transpose(original[i].numpy(), (1, 2, 0)))
            axes[1, i].imshow(np.transpose(corrupted[i].numpy(), (1, 2, 0)))
            axes[2, i].imshow(np.transpose(recon[i].numpy(), (1, 2, 0)))

            axes[0, i].axis("off")
            axes[1, i].axis("off")
            axes[2, i].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=10)
        axes[1, 0].set_ylabel("Corrupt", fontsize=10)
        axes[2, 0].set_ylabel("Recon", fontsize=10)

        plt.suptitle(f"{name} - {corruption} robustness test")
        plt.tight_layout()
        plt.savefig(f"results/{name}_{corruption}_robustness.png")
        plt.close()

        scores = [
            blur_score(np.transpose(img.numpy(), (1, 2, 0)))
            for img in recon[:8]
        ]

        print(f"{name} {corruption} Robustness Blur Score: {np.mean(scores):.6f}")
# --------------------------
# RUN EXPERIMENT
# --------------------------
def run_experiment(name, latent_dim=20, hidden_dim=400, beta=1.0, loss_type="mse"):
    print(f"\nRunning: {name}")
    print(f"latent_dim={latent_dim}, hidden_dim={hidden_dim}, beta={beta}, loss_type={loss_type}")

    model = VAE(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    for epoch in range(EPOCHS):
        loss, recon_loss, kl_loss = train(
            model,
            optimizer,
            beta=beta,
            loss_type=loss_type
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Total Loss: {loss:.4f} | "
            f"Recon Loss: {recon_loss:.4f} | "
            f"KL Loss: {kl_loss:.4f}"
        )

    save_images(model, name)
    measure_inference_time(model, name)
    robustness_test(model, name)

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":

    # Baseline
    run_experiment(
        name="baseline",
        latent_dim=20,
        hidden_dim=400,
        beta=1.0,
        loss_type="mse"
    )

    # Experiment 1: Reconstruction Loss
    run_experiment(
        name="bce_loss",
        latent_dim=20,
        hidden_dim=400,
        beta=1.0,
        loss_type="bce"
    )

    # Experiment 2: KL Strength
    run_experiment(
        name="beta_0.1",
        latent_dim=20,
        hidden_dim=400,
        beta=0.1,
        loss_type="mse"
    )

    run_experiment(
        name="beta_5",
        latent_dim=20,
        hidden_dim=400,
        beta=5.0,
        loss_type="mse"
    )

    # Experiment 3: Latent Dimension
    run_experiment(
        name="latent_10",
        latent_dim=10,
        hidden_dim=400,
        beta=1.0,
        loss_type="mse"
    )

    run_experiment(
        name="latent_50",
        latent_dim=50,
        hidden_dim=400,
        beta=1.0,
        loss_type="mse"
    )

    # Experiment 4: Decoder Capacity
    run_experiment(
        name="big_decoder",
        latent_dim=20,
        hidden_dim=800,
        beta=1.0,
        loss_type="mse"
    )
