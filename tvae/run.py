#from ctgan import CTGAN
import sys
import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  
sys.path.insert(0, root_dir)

from demo import load_demo
from synthesizers.tvae import TVAE


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


tvae = TVAE(
    epochs=100,  
    compress_dims=(256, 256),  # Increase neurons in the encoder
    decompress_dims=(256, 256), # Increase neurons in the decoder
    batch_size=500,
    #verbose = True ERA UN PROBLEMA DE PAQUETE, NO DE TERMINAL Y DEBUG
      
)
print("TVAE is defined in:", TVAE.__module__)


print(hasattr(tvae, 'fit_tvae')) 


tvae.fit_tvae(real_data, discrete_columns)

synthetic_data = tvae.sample(32561)


categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country', 'income']

for col in categorical_columns:
    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)

# Guardar los datos sintéticos corregidos
synthetic_data.to_csv("synthetic_data_tvae.csv", index=False)

