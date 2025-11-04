import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

from variationalVAE import VariationalAutoencoder
from data import train_loader, test_loader  # your existing data.py

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# Hyperparameters
latent_dim = 64
learning_rate = 1e-3
epochs = 50
save_path = "variational_vae.pth"
beta = 0.001  # weight for KL divergence

# Model, Loss, Optimizer
model = VariationalAutoencoder(latent_dim=latent_dim).to(device)
recon_criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float("inf")

# Training function
def train():
    global best_loss
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            # Forward
            outputs, mu, logvar = model(images)

            # Loss
            recon_loss = recon_criterion(outputs, images)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = test()

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with test loss {best_loss:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

# Testing function
def test():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs, mu, logvar = model(images)
            recon_loss = recon_criterion(outputs, images)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Visualization
def visualize_reconstruction(num_images=6):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs, _, _ = model(images)

    images = images.cpu()
    outputs = outputs.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(outputs[i].squeeze(), cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    train()
    print("Training finished.")
    print(f"Best model saved as {save_path}")
    visualize_reconstruction()
