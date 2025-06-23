import sys
import os
import pandas as pd
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from demo import load_demo
from synthesizers.ctgan import CTGAN

N_RUNS = 1000
GENERATOR_DIM = (64, 32)
DISCRIMINATOR_DIM = GENERATOR_DIM
EMBEDDING_DIM = GENERATOR_DIM [1]
BATCH_SIZE = 30
EPOCHS = 300
DEVICE = 'cpu' 
MODEL_NAME = 'CTGAN'
ARCH_NAME = f"{GENERATOR_DIM[0]}-{GENERATOR_DIM[1]}"
OUTPUT_FILE = f"samples_{MODEL_NAME.lower()}_{ARCH_NAME}_{DEVICE}.csv"

real_data = load_demo()
discrete_columns = ['HeartDisease', 'Gender']
real_data = real_data.dropna()

ctgan = CTGAN(
    epochs=EPOCHS,
    embedding_dim=EMBEDDING_DIM,
    generator_dim=GENERATOR_DIM,
    discriminator_dim=DISCRIMINATOR_DIM,
    batch_size=BATCH_SIZE,
    cuda=False
)

ctgan.fit(real_data, discrete_columns)

results = []

for i in range(N_RUNS):
    start_time = time.perf_counter()
    _ = ctgan.sample(2000)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    generation_duration = end_time - start_time

    results.append({
        'Model': MODEL_NAME,
        'Architecture': str(GENERATOR_DIM),
        'Device': DEVICE.upper(),
        'Time_seconds': round(generation_duration, 4)
    })

df = pd.DataFrame(results)
media = df['Time_seconds'].mean()
df = pd.concat([df, pd.DataFrame([{
    'Model': MODEL_NAME,
    'Architecture': str(GENERATOR_DIM),
    'Device': DEVICE.upper(),
    'Time_seconds': round(media, 4)
}])], ignore_index=True)

df.to_csv(OUTPUT_FILE, index=False)
