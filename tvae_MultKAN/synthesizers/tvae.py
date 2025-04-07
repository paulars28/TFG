"""TVAE module."""

import numpy as np
import torch.nn as nn
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_transformer import DataTransformer
print("aqui")
from ctgan.synthesizers.base import BaseSynthesizer, random_state

from kan.KANLayer import KANLayer 
from kan.MultKAN import MultKAN

class Encoder(nn.Module):
    """Encoder modelado completamente por una sola instancia de MultKAN."""

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        
        self.multkan_encoder = MultKAN(width=[data_dim] + list(compress_dims) + [embedding_dim], save_act=True)

        self.fc1 = nn.Linear(embedding_dim, embedding_dim)  # (mu)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)  # (logvar)


    def forward(self, input_):
        """Codifica los datos en una representación latente."""
        feature = self.multkan_encoder(input_)  
        
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(nn.Module):
    """Decoder modelado completamente por una sola instancia de MultKAN."""

    def __init__(self, embedding_dim, compress_dims, data_dim):
        super(Decoder, self).__init__()

        self.multkan_decoder = MultKAN(width=[embedding_dim] + list(reversed(compress_dims)) + [data_dim], save_act=True)
        
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decodifica el espacio latente en datos sintéticos."""
        feature = self.multkan_decoder(input_)  
        return feature, self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    """Función de pérdida: pérdida de reconstrucción + KL Divergence."""
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
                loss.append(nn.functional.cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                ))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    print(sum(loss) * factor / x.size()[0])
    print(KLD / x.size()[0])
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


def save_multkan_model(model, path):
    """Guarda el modelo MultKAN en tres archivos: _state, _config.yml, _cache_data."""
    
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    config = dict(
        width = model.width,
        grid = model.grid,
        k = model.k,
        mult_arity = model.mult_arity,
        base_fun_name = model.base_fun_name,
        symbolic_enabled = model.symbolic_enabled,
        affine_trainable = model.affine_trainable,
        grid_eps = model.grid_eps,
        grid_range = model.grid_range,
        sp_trainable = model.sp_trainable,
        sb_trainable = model.sb_trainable,
        state_id = model.state_id,
        auto_save = model.auto_save,
        ckpt_path = model.ckpt_path,
        round = model.round,
        device = str(model.device)
    )

    for i in range(model.depth):
        config[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

    with open(f'{path}_DIABETES_config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    torch.save(model.state_dict(), f'{path}_stateDIABETES30-300')

    if model.cache_data is not None:
        torch.save(model.cache_data, f'{path}_cache_dataDIABETES30-300')

    print(f"Modelo guardado correctamente en: {path}_state, {path}_config.yml, {path}_cache_data")




class TVAE(BaseSynthesizer):
    """Modelo TVAE basado en MultKAN."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
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

    
    def fit_tvae(self, train_data, discrete_columns=()):

        print("Inicializamos DataTransformer...")
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        print("DataTransformer completado.")

        print("Resumen de variables tras el DataTransformer:")
        print(f"- Dimensión de salida total: {self.transformer.output_dimensions}")
        print(f"- Número de columnas originales: {len(self.transformer.output_info_list)}")
        print(f"- Detalle de cada variable transformada:")

        for i, col_info in enumerate(self.transformer.output_info_list):
            print(f"  Variable {i+1}:")
            for span in col_info:
                tipo = "Discreta (softmax)" if span.activation_fn == 'softmax' else "Continua (tanh)"
                print(f"    - Dimensión: {span.dim}, Activación: {span.activation_fn} ({tipo})")


        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        data_dim = self.transformer.output_dimensions
        print(f"Dimensión de datos de salida: {data_dim}")

        print("Inicializamos Encoder y Decoder")
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self._device)
        print("Encoder y Decoder inicializados correctamente.")

        optimizer = optim.Adam(list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)

        print(f"Entrenamiento iniciado por {self.epochs} épocas con batch size {self.batch_size}...")
        for epoch in range(self.epochs):
            epoch_loss = 0
            print(f"\n[Época {epoch + 1}/{self.epochs}]")


            if epoch < (self.epochs * 3) // 4:
                reg_metric = "edge_forward_spline_n"  
            else:
                reg_metric = "edge_backward"   

            if epoch >= (self.epochs * 2) // 3:
                aplicar_sparsity = True
            else:
                aplicar_sparsity = False  


            for batch_id, data in enumerate(loader):
                optimizer.zero_grad()
                real = data[0].to(self._device)
                print(f"  - Procesando batch {batch_id + 1}/{len(loader)}")
  
                if aplicar_sparsity:

                    # Simulación de MultKAN.forward() en Encoder
                    x = real[:, encoder.multkan_encoder.input_id.long()]
                    encoder.multkan_encoder.cache_data = x

                    encoder_acts = []
                    encoder_acts_premult = []
                    encoder_spline_preacts = []
                    encoder_spline_postacts = []
                    encoder_spline_postsplines = []
                    encoder_acts_scale = []
                    encoder_acts_scale_spline = []
                    encoder_subnode_actscale = []
                    encoder_edge_actscale = []

                    for l in range(encoder.multkan_encoder.depth):
                        x_numerical, preacts, postacts_numerical, postspline = encoder.multkan_encoder.act_fun[l](x)
                        
                        if encoder.multkan_encoder.symbolic_enabled:
                            if x.shape[1] == 0:
                                raise ValueError("x tiene 0 dimensiones en el eje 1, lo que causa un error de indexación en symbolic_fun")
                            x_symbolic, _ = encoder.multkan_encoder.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000)  
                        else:
                            x_symbolic = 0.

                        x = x_numerical + x_symbolic
                        encoder_subnode_actscale.append(torch.std(x, dim=0).detach())
                        x = encoder.multkan_encoder.subnode_scale[l][None, :] * x + encoder.multkan_encoder.subnode_bias[l][None, :]
                        

                        input_range = torch.std(preacts, dim=0) + 0.1
                        output_range = torch.std(postacts_numerical, dim=0)
                        encoder_edge_actscale.append(output_range)
                        encoder_acts_scale.append((output_range / input_range).detach())
                        encoder_acts_scale_spline.append(output_range / input_range)
                        encoder_spline_preacts.append(preacts.detach())
                        encoder_spline_postacts.append(postacts_numerical.detach())
                        encoder_spline_postsplines.append(postspline.detach())

                        encoder_acts_premult.append(x.detach())
                        encoder_acts.append(x.detach())

                    encoder.multkan_encoder.acts = encoder_acts
                    encoder.multkan_encoder.acts_premult = encoder_acts_premult
                    encoder.multkan_encoder.spline_preacts = encoder_spline_preacts
                    encoder.multkan_encoder.spline_postacts = encoder_spline_postacts
                    encoder.multkan_encoder.spline_postsplines = encoder_spline_postsplines
                    encoder.multkan_encoder.acts_scale = encoder_acts_scale
                    encoder.multkan_encoder.acts_scale_spline = encoder_acts_scale_spline
                    encoder.multkan_encoder.subnode_actscale = encoder_subnode_actscale
                    encoder.multkan_encoder.edge_actscale = encoder_edge_actscale
                    

                mu, std, logvar = encoder(real)
                print(f"BATCH: {batch_id +1}--> Mu, std y logvar calculados. Mu: {mu.mean().item():.6f}, std: {std.mean().item():.6f}, logvar: {logvar.mean().item():.6f}")
                    
                if aplicar_sparsity:

                    #  Simulación de MultKAN.forward() en Decoder`
                    decoder_acts = []
                    decoder_acts_premult = []
                    decoder_spline_preacts = []
                    decoder_spline_postacts = []
                    decoder_spline_postsplines = []
                    decoder_acts_scale = []
                    decoder_acts_scale_spline = []
                    decoder_subnode_actscale = []
                    decoder_edge_actscale = []
                    for l in range(self.decoder.multkan_decoder.depth):
                        x_numerical, preacts, postacts_numerical, postspline = self.decoder.multkan_decoder.act_fun[l](x)
                        if self.decoder.multkan_decoder.symbolic_enabled:
                            x_symbolic, _ = self.decoder.multkan_decoder.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000)  
                        else:
                            x_symbolic = 0.

                        x = x_numerical + x_symbolic

                        decoder_subnode_actscale.append(torch.std(x, dim=0).detach())
                        x = self.decoder.multkan_decoder.subnode_scale[l][None, :] * x + self.decoder.multkan_decoder.subnode_bias[l][None, :]
                        

                        input_range = torch.std(preacts, dim=0) + 0.1
                        output_range = torch.std(postacts_numerical, dim=0)
                        decoder_edge_actscale.append(output_range)
                        decoder_acts_scale.append((output_range / input_range).detach())
                        decoder_acts_scale_spline.append(output_range / input_range)
                        decoder_spline_preacts.append(preacts.detach())
                        decoder_spline_postacts.append(postacts_numerical.detach())
                        decoder_spline_postsplines.append(postspline.detach())

                        decoder_acts_premult.append(x.detach())
                        decoder_acts.append(x.detach())

                    self.decoder.multkan_decoder.acts = decoder_acts
                    self.decoder.multkan_decoder.acts_premult = decoder_acts_premult
                    self.decoder.multkan_decoder.spline_preacts = decoder_spline_preacts
                    self.decoder.multkan_decoder.spline_postacts = decoder_spline_postacts
                    self.decoder.multkan_decoder.spline_postsplines = decoder_spline_postsplines                    
                    self.decoder.multkan_decoder.acts_scale = decoder_acts_scale
                    self.decoder.multkan_decoder.acts_scale_spline = decoder_acts_scale_spline
                    self.decoder.multkan_decoder.subnode_actscale = decoder_subnode_actscale
                    self.decoder.multkan_decoder.edge_actscale = decoder_edge_actscale
                
                eps = torch.randn_like(mu)
                z = mu + eps * std
                rec, sigmas = self.decoder(z)
                print(f"BATCH: {batch_id + 1}--> Rec y sigmas calculados. Rec: {rec.mean().item():.6f}, sigmas: {sigmas.mean().item():.6f} ")



                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2

            
                if aplicar_sparsity:

                    encoder.multkan_encoder.attribute()
                    self.decoder.multkan_decoder.attribute()

                    sparsity_encoder = encoder.multkan_encoder.get_reg(reg_metric, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.001, lamb_coefdiff=0.001)
                    sparsity_decoder = self.decoder.multkan_decoder.get_reg(reg_metric, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.001, lamb_coefdiff=0.001)

                    sparsity_penalty = 0.001 * (sparsity_encoder + sparsity_decoder)
                    
                    loss = loss + sparsity_penalty
                    print(f"SPARSE: sparsity_penalty: {sparsity_penalty}, sparsity_encoder: {sparsity_encoder} ({sparsity_encoder * 0.001}),  sparsity_decoder: {sparsity_decoder}, ({sparsity_decoder * 0.001})" )
                    print(f"loss + sparsity_penalty = {loss} ")
    
                print(f"loss = {loss}")
                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 2.0)
                epoch_loss += loss.detach().cpu().item()
            
           

        print("Entrenamiento completado.")
        encoder.multkan_encoder.acts = encoder_acts
        encoder.multkan_encoder.acts_premult = encoder_acts_premult
        encoder.multkan_encoder.spline_preacts = encoder_spline_preacts
        encoder.multkan_encoder.spline_postacts = encoder_spline_postacts
        encoder.multkan_encoder.spline_postsplines = encoder_spline_postsplines
        encoder.multkan_encoder.acts_scale = encoder_acts_scale
        encoder.multkan_encoder.acts_scale_spline = encoder_acts_scale_spline
        encoder.multkan_encoder.subnode_actscale = encoder_subnode_actscale
        encoder.multkan_encoder.edge_actscale = encoder_edge_actscale
        self.decoder.multkan_decoder.acts = decoder_acts
        self.decoder.multkan_decoder.acts_premult = decoder_acts_premult
        self.decoder.multkan_decoder.spline_preacts = decoder_spline_preacts
        self.decoder.multkan_decoder.spline_postacts = decoder_spline_postacts
        self.decoder.multkan_decoder.spline_postsplines = decoder_spline_postsplines                    
        self.decoder.multkan_decoder.acts_scale = decoder_acts_scale
        self.decoder.multkan_decoder.acts_scale_spline = decoder_acts_scale_spline
        self.decoder.multkan_decoder.subnode_actscale = decoder_subnode_actscale
        self.decoder.multkan_decoder.edge_actscale = decoder_edge_actscale
        encoder.multkan_encoder.cache_data = encoder_acts
        self.decoder.multkan_decoder.cache_data = decoder_acts



        print("Guardando encoder y decoder tras entrenamiento")
        print(f"Dimensiones ENCODER antes de prune_node: {encoder.multkan_encoder.width}")
        print(f"Dimensiones DECODER antes de prune_node: {self.decoder.multkan_decoder.width}")
        save_multkan_model(encoder.multkan_encoder, "./MODELOS/final_encoder")
        save_multkan_model(self.decoder.multkan_decoder, "./MODELOS/final_decoder")

        print("Guardado completado.")

               
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
