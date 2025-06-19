
"""CARGAMOS SOLO DECODER"""

import torch
import sys
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  
sys.path.insert(0, root_dir)


from demo import load_demo


from data_transformer import DataTransformer
from kan.MultKAN import MultKAN  

decoder_path = "/home/gtav-tft/Desktop/SYNTHEMA/TVAE/CTGAN_MultKANJ/MODELOS/prueba3/final_decoder_state"
decoder_cache="/home/gtav-tft/Desktop/SYNTHEMA/TVAE/CTGAN_MultKANJ/MODELOS/prueba3/final_decoder_cache_data"



ddata_dim = 147  
dcompress_dims = [128, 128] 
dembedding_dim = 128

from ctgan.synthesizers.tvae import Decoder  

decoder = Decoder(dembedding_dim, dcompress_dims, ddata_dim).to('cuda')
decoder.multkan_decoder.load_state_dict(torch.load(decoder_path), strict=True)
decoder.multkan_decoder.cache_data = torch.load(decoder_cache).to('cuda')
decoder.eval()  


def sample_only_decoder(samples, decoder, transformer, device='cuda', batch_size=2500, embedding_dim=128):
    decoder.eval()
    decoder.to(device)

    steps = samples // batch_size + 1
    data = []

    for _ in range(steps):
        noise = torch.randn(batch_size, embedding_dim).to(device)
        fake, sigmas = decoder(noise) 
        fake = torch.tanh(fake)        
        data.append(fake.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)[:samples]
    return transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

real_data = load_demo()
real_data.drop(columns=['fnlwgt'], inplace=True)

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]
transformer = DataTransformer()
transformer = DataTransformer()
transformer.fit(real_data, discrete_columns)
train_data = transformer.transform(real_data)
synthetic_data = sample_only_decoder( 32561, decoder, transformer)
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country', 'income']
for col in categorical_columns:
    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)
# Guardar los datos sintéticos generados
os.makedirs("./SAMPLE", exist_ok=True)
synthetic_data.to_csv("./SAMPLE/synthetic_data_tvaeKAN.csv", index=False)

