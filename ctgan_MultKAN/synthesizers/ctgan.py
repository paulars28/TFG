"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
import yaml
import os

from ctgan.data_sampler import DataSampler
from data_transformer import DataTransformer
from errors import InvalidDataError
from synthesizers.base import BaseSynthesizer, random_state
from matplotlib import pyplot as plt

from kan.KANLayer import KANLayer 
from kan.MultKAN import MultKAN


#función para guardar el modelo
def save_multkan_model(model, path_base, nombre_modelo="modelo"):

    directory = os.path.dirname(path_base)
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

    # Guardar archivos
    path_config = f"{path_base}_config_{nombre_modelo}.yml"
    path_state = f"{path_base}_state_{nombre_modelo}.pt"
    path_cache = f"{path_base}_cache_data_{nombre_modelo}.pt"

    with open(path_config, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    torch.save(model.state_dict(), path_state)

    if model.cache_data is not None:
        torch.save(model.cache_data, path_cache)
        print(f"[✓] Modelo y cache guardados: {path_state}, {path_config}, {path_cache}")
    else:
        print(f"[✓] Modelo guardado sin cache: {path_state}, {path_config}")


def cache_multkan_activations(multkan_module, x):
    multkan_module.cache_data = x
    acts = []; acts_premult = []; spline_preacts = []; spline_postacts = []
    spline_postsplines = []; acts_scale = []; acts_scale_spline = []
    subnode_actscale = []; edge_actscale = []

    for l in range(multkan_module.depth):
        x_num, preacts, postacts_num, postspline = multkan_module.act_fun[l](x)
        x_symb, _ = multkan_module.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000) if multkan_module.symbolic_enabled else (0., None)
        x = x_num + x_symb
        subnode_actscale.append(torch.std(x, dim=0).detach())
        x = multkan_module.subnode_scale[l][None, :] * x + multkan_module.subnode_bias[l][None, :]
        input_range = torch.std(preacts, dim=0) + 0.1
        output_range = torch.std(postacts_num, dim=0)
        edge_actscale.append(output_range)
        acts_scale.append((output_range / input_range).detach())
        acts_scale_spline.append(output_range / input_range)
        spline_preacts.append(preacts.detach())
        spline_postacts.append(postacts_num.detach())
        spline_postsplines.append(postspline.detach())
        acts_premult.append(x.detach())
        acts.append(x.detach())

    multkan_module.acts = acts
    multkan_module.acts_premult = acts_premult
    multkan_module.spline_preacts = spline_preacts
    multkan_module.spline_postacts = spline_postacts
    multkan_module.spline_postsplines = spline_postsplines
    multkan_module.acts_scale = acts_scale
    multkan_module.acts_scale_spline = acts_scale_spline
    multkan_module.subnode_actscale = subnode_actscale
    multkan_module.edge_actscale = edge_actscale

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        
        """
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)
        """

        self.discriminator_multkan= MultKAN( width=[self.pacdim] + list(discriminator_dim) + [1], save_act=False, symbolic_enabled=False)


    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.discriminator_multkan(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        
        """
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)"""

        self.generator_multkan = MultKAN(width=[embedding_dim] + list(generator_dim) + [data_dim], save_act=False, symbolic_enabled=False)


    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.generator_multkan(input_)
        return data
        
def get_variables_transformer(transformer):
    mapping = {}
    count = 0
    print(transformer._column_transform_info_list)
    for info in transformer._column_transform_info_list:
        col = info.column_name
        dim = info.output_dimensions
        for i in range(dim):
            nombre_real = f"{col}" if dim == 1 else f"{col}_{i}"
            mapping[f"x_{count+1}"] = nombre_real
            count += 1
    print(f"Variables transformadas: {mapping}")

    with open ("variables_transformer.txt", "w") as f:
        f.write("Variables transformadas:\n")
        for key, value in mapping.items():
            f.write(f"{key} -> {value}\n")

 
    

class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        #embedding_dim=128,
        #generator_dim=(256, 256),
        #discriminator_dim=(256, 256),
        embedding_dim=32,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=5e-5,

        discriminator_decay=1e-6,
        batch_size=32,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=100,
        pac=10,
        cuda=True,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        #self._validate_discrete_columns(train_data, discrete_columns)
        #self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        print("Inicializamos DataTransformer...")
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        print("DataTransformer completado.")


        data_dim = self._transformer.output_dimensions
        print(f"Dimensiones de los datos: {data_dim}")
        print(1)
        get_variables_transformer(self._transformer)
        print(2)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        print(self._transformer.output_info_list)
        print(f"Data_dim = {self._transformer.output_dimensions}")
        


        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        self.discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        print("Generator width:", self._generator.generator_multkan.width)
        print("Discriminator width:", self.discriminator.discriminator_multkan.width)


        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            self.discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        self.training_metrics = []  

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:


            #modelamos parámetros para el sparsity
            if i < (self._epochs * 3) // 4:
                reg_metric = "edge_forward_spline_n"  
            else:
                reg_metric = "edge_backward"   
            if i >= (self._epochs * 7) // 10:
                aplicar_sparsity = True
            else:
                aplicar_sparsity = True  


            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):

                    
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self.discriminator(fake_cat)
                    y_real = self.discriminator(real_cat)

                    pen = self.discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    y_fake = self.discriminator(fake_cat)
                    y_real = self.discriminator(real_cat)

                    pen = self.discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))



                    if aplicar_sparsity:
                        x_discrim = fake_cat.detach()
                        if x_discrim.shape[0] % self.pac != 0:
                            continue
                        x_discrim_pac = x_discrim.view(-1, self.discriminator.pacdim)
                        cache_multkan_activations(self.discriminator.discriminator_multkan, x_discrim_pac)

                        """x_discrim = fake_cat.detach().view(-1, self.discriminator.pacdim // self.pac)
                        cache_multkan_activations(self.discriminator.discriminator_multkan, x_discrim)"""

                    if aplicar_sparsity:
                        self.discriminator.discriminator_multkan.attribute()
                        sparsity_d = self.discriminator.discriminator_multkan.get_reg(reg_metric, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.005, lamb_coefdiff=0.001)
                        loss_d += 0.001 * sparsity_d
                        print(f"SPARSE: sparsity_penalty_discriminator: {sparsity_d}")
                        print(f"loss + sparsity_penalty = {loss_d} ")


                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    print(i)

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy


                fakeact = self._apply_activate(fake)

                if condvec is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy


            # === Cacheo de activaciones GENERATOR ===
            if aplicar_sparsity:
                x_gen = fakez.detach()
                cache_multkan_activations(self._generator.generator_multkan, x_gen)

            if aplicar_sparsity:
                self._generator.generator_multkan.attribute()
                sparsity_g = self._generator.generator_multkan.get_reg(reg_metric, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.001, lamb_coefdiff=0.001)
                loss_g += 0.001 * sparsity_g
                print(f"SPARSE: sparsity_penalty_generator: {sparsity_g}")
                print(f"loss + sparsity_penalty = {loss_g} ")

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
                
        self.loss_values.to_csv("ctgan_training_metrics.csv", index=False)
        df = self.loss_values
        plt.plot(df['Epoch'], df['Generator Loss'], label='Generator Loss')
        plt.plot(df['Epoch'], df['Discriminator Loss'], label='Discriminator Loss')
        if 'Sparsity Penalty' in df.columns:
            plt.plot(df['Epoch'], df['Sparsity Penalty'], label='Sparsity')
        plt.legend(); plt.grid(); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("CTGAN Training Metrics")
        plt.tight_layout()
        plt.savefig("ctgan_training_metrics.png")

        gen_mean = df['Generator Loss'].mean()
        dis_mean = df['Discriminator Loss'].mean()
        gen_std = df['Generator Loss'].std()
        dis_std = df['Discriminator Loss'].std()

        # Gráfica
        plt.figure(figsize=(10,6))
        plt.plot(df['Epoch'], df['Generator Loss'], label='Generator Loss', linewidth=1.5)
        plt.plot(df['Epoch'], df['Discriminator Loss'], label='Discriminator Loss', linewidth=1.5)

        # Líneas horizontales de media
        plt.axhline(gen_mean, color='blue', linestyle='--', alpha=0.5, label='Gen Mean')
        plt.axhline(dis_mean, color='orange', linestyle='--', alpha=0.5, label='Dis Mean')

        # Relleno para desviación típica (±σ)
        plt.fill_between(df['Epoch'], gen_mean - gen_std, gen_mean + gen_std, color='blue', alpha=0.1)
        plt.fill_between(df['Epoch'], dis_mean - dis_std, dis_mean + dis_std, color='orange', alpha=0.1)

        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.title("Evolución de pérdidas en el entrenamiento de CTGAN")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ctgan_loss_analysis.png")
        plt.show()


        save_multkan_model(self._generator.generator_multkan, "./MODELOS/final_generator")
        save_multkan_model(self.discriminator.discriminator_multkan, "./MODELOS/final_discriminator")

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
