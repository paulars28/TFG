import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kan.KANLayer import KANLayer 


# ------------------------------
# Configuración
# ------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LATENT_DIM = 64

# ------------------------------
# Dataset
# ------------------------------
class DiabetesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------------
# Modelo VAE con KAN ajustado
# ------------------------------
class VAE_KAN(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super(VAE_KAN, self).__init__()
        self.kan_enc = KANLayer(input_dim, 128)
        self.fc_mu = KANLayer(128, latent_dim)
        self.fc_logvar = KANLayer(128, latent_dim)
        self.kan_dec = KANLayer(latent_dim, 128)
        self.out = KANLayer(128, input_dim)

    def encode(self, x):
        h, *_ = self.kan_enc(x)
        mu, *_ = self.fc_mu(h)
        logvar, *_ = self.fc_logvar(h)
        return mu, logvar

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h, *_ = self.kan_dec(z)
        out, *_ = self.out(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        return self.decode(z), mu, logvar

# ------------------------------
# Pérdida
# ------------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x[:, :-1], x[:, :-1], reduction='mean')
    bce_loss = F.binary_cross_entropy_with_logits(recon_x[:, -1], x[:, -1], reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + bce_loss + kld, recon_loss, bce_loss, kld

# ------------------------------
# Datos y preprocesado
# ------------------------------
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_all = np.concatenate([X_scaled, y.values.reshape(-1, 1)], axis=1)
X_train, X_test = train_test_split(X_all, test_size=0.2, random_state=42)

train_loader = DataLoader(DiabetesDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# Entrenamiento
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE_KAN(input_dim=X_all.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, recon_l, bce_l, kld_l = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Recon: {recon_l:.4f} | BCE: {bce_l:.4f} | KLD: {kld_l:.4f}")

# ------------------------------
# Generación de datos
# ------------------------------
model.eval()
z = torch.randn(len(df), LATENT_DIM).to(device)
with torch.no_grad():
    logits = model.decode(z).cpu()
    outcome_probs = torch.sigmoid(logits[:, -1])
    generated_outcome = (outcome_probs > 0.5).int().numpy().reshape(-1, 1)
    logits_np = logits[:, :-1].numpy()
    generated_features = logits_np * scaler.scale_ + scaler.mean_

# Redondear características según tipo original
real_types = df.drop(columns=["Outcome"]).dtypes
for i, col in enumerate(X.columns):
    if np.all(df[col] == df[col].astype(int)):
        generated_features[:, i] = np.where(generated_features[:, i] % 1 >= 0.5,
                                            np.ceil(generated_features[:, i]),
                                            np.floor(generated_features[:, i]))
        generated_features[:, i] = generated_features[:, i].astype(int)
    else:
        generated_features[:, i] = np.round(generated_features[:, i], 1)

# Concatenar y ajustar columna Outcome
synthetic_data = np.concatenate([generated_features, generated_outcome], axis=1)
synthetic_df = pd.DataFrame(synthetic_data, columns=list(X.columns) + ["Outcome"])

for col in X.columns:
    if np.all(df[col] == df[col].astype(int)):
        synthetic_df[col] = synthetic_df[col].astype(int)
    else:
        synthetic_df[col] = synthetic_df[col].round(1)

synthetic_df["Outcome"] = synthetic_df["Outcome"].astype(int)

# Guardar
synthetic_df.to_csv("synthetic_diabetes_kan.csv", index=False)

# ------------------------------
# Comparación y visualización
# ------------------------------
def plot_distributions(real_data, synthetic_data, columns):
    for i, name in enumerate(columns):
        plt.figure(figsize=(6, 4))
        sns.histplot(real_data[name], color="blue", label="Real", stat="density", kde=True, alpha=0.6)
        sns.histplot(synthetic_data[:, i], color="orange", label="Synthetic", stat="density", kde=True, alpha=0.6)
        plt.title(f"Distribución: {name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

columns = list(X.columns) + ["Outcome"]
real_data_unscaled = pd.concat([X, y], axis=1)
#plot_distributions(real_data_unscaled, synthetic_data, columns)

print("\n\U0001F4CA MSE por feature:")
for i, name in enumerate(columns[:-1]):
    mse = mean_squared_error(real_data_unscaled[name], synthetic_data[:, i])
    print(f" - {name}: MSE = {mse:.4f}")

print(f"\n✅ Outcome Real % positivos: {real_data_unscaled['Outcome'].mean():.2%}")
print(f"✅ Outcome Sintético % positivos: {synthetic_data[:, -1].mean():.2%}")