import os
import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, accuracy_score, roc_auc_score, precision_score, recall_score
from scipy.stats import entropy, ks_2samp, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
import warnings
from matplotlib import pyplot as plt

list = [
    "/home/gtav-tft/Desktop/paula/eval/train_metrics_tvae_MultKAN/10-5/tvaeKAN_training_metrics.csv",
    "/home/gtav-tft/Desktop/paula/eval/train_metrics_tvae_MultKAN/16-8/tvaeKAN_training_metrics.csv",
    "/home/gtav-tft/Desktop/paula/eval/train_metrics_tvae_MultKAN/32-16/tvaeKAN_training_metrics.csv",
    "/home/gtav-tft/Desktop/paula/eval/train_metrics_tvae_MultKAN/64-32/tvaeKAN_training_metrics.csv"
]

names = ['10-5', '16-8', '32-16', '64-32']
dfs = [pd.read_csv(path) for path in list]


linestyles = {
    'total_loss': 'solid',
    'recon_loss': 'dashed',
    'kl_loss': 'dotted'
}

colors = plt.get_cmap('tab10') 

plt.figure(figsize=(12, 8))

for idx, (df, name) in enumerate(zip(dfs, names)):
    color = colors(idx)
    for loss_type, style in linestyles.items():
        #df_limitada=df[df['total_loss'] <= 25]
        plt.plot(df['epoch'], df[loss_type],
                label=f'{name} {loss_type}',
                linestyle=style,
                color=color,
                marker='o' if loss_type == 'total_loss' else None)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 25) 
plt.title('Comparación de pérdida por época en TVAE-KAN por arquitectura')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tvae_loss_comparison_kan.png")
plt.show()
