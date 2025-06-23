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
#real_data.drop(columns=['QuantumPatternFeature'], inplace=True)
real_data.dropna()


discrete_columns = [
    'HeartDisease',
    'Gender'

]

real_data = real_data.dropna()  

tvae = TVAE(
    epochs=100,  
    compress_dims=(32, 16),
    embedding_dim=16,
    batch_size=32,    
)



tvae.fit_tvae(real_data, discrete_columns)



synthetic_data = tvae.sample(2000)

categorical_columns = [
    'HeartDisease',
    'Gender'

]
'''
for col in categorical_columns:
    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)'''



synthetic_data.to_csv("/home/gtav-tft/Desktop/paula/eval/DATASETS_EVALUATION/synthetic_heartdisease_mlp.csv", index=False)
output_txt = "/home/gtav-tft/Desktop/paula/eval/DATASETS_EVALUATION/synthetic_heartdisease_mlp_params.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("Parámetros del modelo TVAE utilizados:\n")
    f.write(f"epochs = {tvae.epochs}\n")
    f.write(f"compress_dims = {tvae.compress_dims}\n")
    f.write(f"embedding_dim = {tvae.embedding_dim}\n")
    f.write(f"batch_size = {tvae.batch_size}\n")
    f.write(f"discrete_columns = {discrete_columns}\n")
    f.write(f"n_sampled_rows = 20000\n")
    f.write(f"input_data_shape = {real_data.shape}\n")

