"""TVAE module."""

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from synthesizers.base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE. """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE."""

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
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
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """TVAE."""
    print("Custom TVAE initialized!")

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

    @random_state
    def fit_tvae(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data."""

        print("Initializing data transformer...", flush=True)
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        print(f"Training started for {self.epochs} epochs with batch size {self.batch_size}...", flush=True)
        self.training_metrics = []  


        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_loss_values = []

            print(f"Epoch {epoch + 1}/{self.epochs}", flush=True)
            for batch_id, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch_loss = loss.detach().cpu().item()
                batch_loss_values.append(batch_loss)
                epoch_loss += batch_loss

                # Print batch progress
                #print(f"  Batch {batch_id + 1}/{len(loader)} - Loss: {batch_loss:.4f}", flush=True)
            epoch_recon_loss = loss_1.detach().cpu().item()
            epoch_kl_loss = loss_2.detach().cpu().item()

            self.training_metrics.append({
                "epoch": epoch + 1,
                "total_loss": epoch_loss / len(loader),
                "recon_loss": epoch_recon_loss,
                "kl_loss": epoch_kl_loss,
            })

            # Save epoch loss to DataFrame
            epoch_loss_avg = epoch_loss / len(loader)
            self.loss_values = pd.concat(
                [self.loss_values, pd.DataFrame([{'Epoch': epoch + 1, 'Loss': epoch_loss_avg}])],
                ignore_index=True
            )

            # Print epoch summary
            print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss_avg:.4f}\n", flush=True)

            metrics_df = pd.DataFrame(self.training_metrics)
            metrics_df.to_csv("tvae_training_metrics.csv", index=False)

            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df["epoch"], metrics_df["total_loss"], label="Total Loss")
            plt.plot(metrics_df["epoch"], metrics_df["recon_loss"], label="Reconstruction Loss")
            plt.plot(metrics_df["epoch"], metrics_df["kl_loss"], label="KL Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("TVAE Training Metrics")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("tvae_training_metrics.png")  # también puedes guardar como PDF
            plt.close()


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

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
