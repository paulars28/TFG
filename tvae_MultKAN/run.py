#from ctgan import CTGAN
import sys
import os
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  
sys.path.insert(0, root_dir)

from demo import load_demo
from synthesizers.tvae import TVAE


real_data = load_demo()
#Real_data.drop(columns=['fnlwgt'], inplace=True)

'''discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]'''

discrete_columns = [
    'Outcome'
]

real_data = real_data.dropna()  



tvae = TVAE(
    epochs=300,
    embedding_dim=30,
    compress_dims=(30, 30),
    batch_size=1800, 
)

print("TVAE is defined in:", TVAE.__module__)
print(hasattr(tvae, 'fit_tvae')) 


#tvae.fit_tvae(real_data, discrete_columns)
tvae.fit_tvae(real_data, discrete_columns)
torch.cuda.empty_cache()


# Combinar todos los lotes en un solo dataset
synthetic_data = tvae.sample(32561)
#categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
#                        'relationship', 'race', 'sex', 'native-country', 'income']
#for col in categorical_columns:
#    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)
# Guardar los datos sintéticos generados
categorical_columns= [
    'Outcome'
]
for col in categorical_columns:
    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)
synthetic_data.to_csv("synthetic_data_tvaeKANDIABETES30-300.csv", index=False)





