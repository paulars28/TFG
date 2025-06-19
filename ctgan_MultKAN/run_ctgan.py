#from ctgan import CTGAN
import sys
import os
import pandas as pd
from torchsummary import summary


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





ctgan.fit(real_data, discrete_columns=discrete_columns)

print("CTGAN GENERATOR:")
print(ctgan._generator)
print("CTGAN DISCRIMINATOR:")
print(ctgan.discriminator)
print("CTGAN Generator MultKAN:")
print(ctgan._generator.generator_multkan)
print("CTGAN Discriminator MultKAN:")
print(ctgan.discriminator.discriminator_multkan)

print("Generator MULTKAN parameters:")
for name, param in ctgan._generator.generator_multkan.named_parameters():
    print(f"Generator - {name}: {param.shape}")

print("Discriminator MULTKAN parameters:")
for name, param in ctgan.discriminator.discriminator_multkan.named_parameters():
    print(f"Discriminator - {name}: {param.shape}")

for col, info in zip(real_data.columns, ctgan._transformer.output_info_list):
    print(f"{col}: {info}")



synthetic_data = ctgan.sample(2000)
synthetic_data.to_csv("/home/gtav-tft/Desktop/paula/eval_ctgan/DATASETS_EVALUATION/synthetic_heartdisease_kan.csv", index=False)





