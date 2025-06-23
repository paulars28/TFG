import time
import pandas as pd
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from demo import load_demo
from synthesizers.tvae import TVAE

real_data = load_demo()
real_data = real_data.dropna()
discrete_columns = ['HeartDisease', 'Gender']

N_RUNS = 1000
ARCHITECTURE = (64, 32)
BATCH_SIZE = 32
DEVICE = 'cpu'
MODEL_NAME = 'TVAE'
OUTPUT_FILE = f"samples_{MODEL_NAME.lower()}_{ARCHITECTURE[0]}-{ARCHITECTURE[1]}_{DEVICE}.csv"

tvae = TVAE(
    epochs=100,
    compress_dims=ARCHITECTURE,
    embedding_dim=ARCHITECTURE[1],
    batch_size=BATCH_SIZE,
    cuda=False
)

tvae.fit_tvae(real_data, discrete_columns)

results = []

for i in range(N_RUNS):
    start_time = time.perf_counter()
    _ = tvae.sample(2000)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    generation_duration = end_time - start_time

    results.append({
        'Model': MODEL_NAME,
        'Architecture': str(ARCHITECTURE),
        'Device': DEVICE.upper(),
        'Time_seconds': round(generation_duration, 4)
    })

df = pd.DataFrame(results)
media = df['Time_seconds'].mean()
df = pd.concat([df, pd.DataFrame([{
    'Model': MODEL_NAME,
    'Architecture': str(ARCHITECTURE),
    'Device': DEVICE.upper(),
    'Time_seconds': round(media, 4)
}])], ignore_index=True)

df.to_csv(OUTPUT_FILE, index=False)
