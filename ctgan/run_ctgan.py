#from ctgan import CTGAN
import sys
import os
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  
sys.path.insert(0, root_dir)

from demo import load_demo
from synthesizers.ctgan import CTGAN



real_data = load_demo()
#real_data.drop(columns=['QuantumPatternFeature'], inplace=True)
real_data.dropna()


discrete_columns = [
    'HeartDisease',
    'Gender'

]

real_data = real_data.dropna()  

ctgan = CTGAN(
    epochs=300,  
        embedding_dim=5,
        generator_dim=(10, 5),
        discriminator_dim=(10, 5),
    batch_size=30,    
)

ctgan.fit(real_data, discrete_columns)



synthetic_data = ctgan.sample(2000)

categorical_columns = [
    'HeartDisease',
    'Gender'

]
'''
for col in categorical_columns:
    synthetic_data[col] = synthetic_data[col].apply(lambda x: f" {x}" if pd.notna(x) else x)'''



synthetic_data.to_csv("/home/gtav-tft/Desktop/paula/eval_ctgan/DATASETS_EVALUATION/synthetic_heartdisease_mlp.csv", index=False)

