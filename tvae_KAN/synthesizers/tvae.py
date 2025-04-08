"""TVAE module."""

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_transformer import DataTransformer
print("aqui")
from synthesizers.base import BaseSynthesizer, random_state

from kan.KANLayer import KANLayer  


class Encoder(nn.Module):
    """Encoder for TVAE using only KAN layers."""

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        
        # Using KAN layers for entire encoder transformation
        self.kan1 = KANLayer(in_dim=data_dim, out_dim=compress_dims[0])  # First KAN layer
        self.kan2 = KANLayer(in_dim=compress_dims[0], out_dim=compress_dims[1])  # Second KAN layer
        
        # Final fully connected layers to compute latent distribution
        self.fc1 = torch.nn.Linear(compress_dims[1], embedding_dim)  # Mean (mu)
        self.fc2 = torch.nn.Linear(compress_dims[1], embedding_dim)  # Log variance (logvar)

    def forward(self, input_):
        """Encode input data using KAN transformations."""
        feature, _, _, _ = self.kan1(input_)
        feature, _, _, _ = self.kan2(feature)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(nn.Module):
    """Decoder for TVAE using only KAN layers."""

    def __init__(self, embedding_dim, compress_dims, data_dim):
        super(Decoder, self).__init__()

        # Using KAN layers for entire decoding process
        self.kan1 = KANLayer(in_dim=embedding_dim, out_dim=compress_dims[1])  # First KAN layer
        self.kan2 = KANLayer(in_dim=compress_dims[1], out_dim=compress_dims[0])  # Second KAN layer
        
        # Final KAN layer to reconstruct original data
        self.kan_out = KANLayer(in_dim=compress_dims[0], out_dim=data_dim)
        
        # Learnable standard deviation for reconstruction
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode latent space into synthetic data using KAN layers."""
        feature, _, _, _ = self.kan1(input_)
        feature, _, _, _ = self.kan2(feature)
        feature, _, _, _ = self.kan_out(feature)
        return feature, self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    """Compute the loss function: Reconstruction Loss + KL Divergence."""
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            else:
                ed = st + span_info.dim
                loss.append(torch.nn.functional.cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                ))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """Fully KAN-based TVAE model."""

    print("TVAE-KAN Model Initialized!")

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),    # Neurons in each KAN layer 
        l2scale=1e-5,
        batch_size=500,
        epochs=100,
        loss_factor=2,
        cuda=True,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        else:
            device = 'cuda'

        self._device = torch.device(device)

    @random_state
    def fit_tvae(self, train_data, discrete_columns=()):
        """Train the TVAE-KAN model."""
        print("Initializing data transformer...", flush=True)
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self._device)

        optimizer = Adam(list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)

        print(f"Training started for {self.epochs} epochs with batch size {self.batch_size}...", flush=True)

        for epoch in range(self.epochs):
            epoch_loss = 0

            for batch_id, data in enumerate(loader):
                optimizer.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                epoch_loss += loss.detach().cpu().item()

            epoch_loss_avg = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss_avg:.4f}\n", flush=True)

        print("Training completed.", flush=True)

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
