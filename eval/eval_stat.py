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
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency, chisquare, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
import warnings
from scipy.spatial import distance

target = "HeartDisease" 
arquitectura= "64-32" 


# Clase para redirigir stdout a la consola y a un archivo
class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def preprocess_data(real_data, tvae_data, kan_data, numerical_columns, categorical_columns):
    scaler = MinMaxScaler()
    le_dict = {}
    real_data[numerical_columns] = scaler.fit_transform(real_data[numerical_columns])
    tvae_data[numerical_columns] = scaler.transform(tvae_data[numerical_columns])
    kan_data[numerical_columns] = scaler.transform(kan_data[numerical_columns])

    for col in categorical_columns:
        le = LabelEncoder()
        real_data[col] = le.fit_transform(real_data[col].astype(str))
        tvae_data[col] = le.transform(tvae_data[col].astype(str))
        kan_data[col] = le.transform(kan_data[col].astype(str))
        le_dict[col] = le
    return real_data, tvae_data, kan_data

def student_t_tests(numerical_columns, real, synthetic): 
    p_values = []
    
    for c in numerical_columns:
        _, p = ttest_ind(real[c], synthetic[c])
        p_values.append(p)
    for i, p in enumerate(p_values):
        print(f"Column {numerical_columns[i]}: p-value = {p:.4f}")
        

def mann_whitney_tests(numerical_columns, real, synthetic):
    p_values = []
    
    for c in numerical_columns:
        _, p = mannwhitneyu(real[c], synthetic[c])
        p_values.append(p)
    for i, p in enumerate(p_values):
        print(f"Column {numerical_columns[i]}: p-value = {p:.4f}")


def ks_tests(numerical_columns, real, synthetic):
    p_values = []
    
    for c in numerical_columns:
        _, p = ks_2samp(real[c], synthetic[c])
        p_values.append(p)
    for i, p in enumerate(p_values):
        print(f"Column {numerical_columns[i]}: p-value = {p:.4f}")


def chi_squared_tests(categorical_columns, real, synthetic):
    p_values = []
    
    for c in categorical_columns:
        contingency_table = pd.crosstab(real[c], synthetic[c])
        _, p, _, _ = chi2_contingency(contingency_table)
        p_values.append(p)
    for i, p in enumerate(p_values):
        print(f"Column {categorical_columns[i]}: p-value = {p:.4f}")


def cosine_distances(numerical_columns, real, synthetic):
    dists = []
    for c in numerical_columns:  
        dists.append(distance.cosine(real[c].values, synthetic[c].values))
    for i, p in enumerate(dists):
        print(f"Column {numerical_columns[i]}: Cosine distance = {p:.4f}")

def js_distances(numerical_columns, real, synthetic, bins=50):
    dists = []

    for c in numerical_columns:
   
        real_hist, bin_edges = np.histogram(real[c], bins=bins, range=(min(real[c].min(), synthetic[c].min()), 
                                                                        max(real[c].max(), synthetic[c].max())), density=True)
        synth_hist, _ = np.histogram(synthetic[c], bins=bin_edges, density=True)

        epsilon = 1e-12
        real_hist += epsilon
        synth_hist += epsilon

        real_prob = real_hist / np.sum(real_hist)
        synth_prob = synth_hist / np.sum(synth_hist)

        js = distance.jensenshannon(real_prob, synth_prob)
        dists.append(js)

        print(f"Column {c}: Jensen-Shannon distance = {js:.4f}")

    return dists

def wass_distances(numerical_columns, real, synthetic):
    dists = []
    for c in numerical_columns:  
        dists.append(wasserstein_distance(real[c].values, synthetic[c].values))
    for i, p in enumerate(dists):
        print(f"Column {numerical_columns[i]}: Wasserstein distance = {p:.4f}")

    

if __name__ == "__main__":

    
    os.makedirs("dbeval", exist_ok=True)
    sys.stdout = DualOutput("/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/results_stats.txt")

    real_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS_EVALUATION/Heart Prediction Quantum Dataset.csv"
    tvae_path = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/synthetic_heartdisease_mlp.csv"
    kan_path = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/synthetic_heartdisease_MultKAN.csv"
    #tvae_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS_EVALUATION/synthetic_heartdisease_mlp.csv"
    #kan_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS_EVALUATION/synthetic_heartdisease_MultKAN.csv"

    categorical_columns= ['HeartDisease','Gender']
    numerical_columns = ['Age','BloodPressure','HeartRate','QuantumPatternFeature',]

    real_data, tvae_data, kan_data = preprocess_data(
        pd.read_csv(real_path).sample(500, random_state=42),
        pd.read_csv(tvae_path).sample(500,  random_state=42),
        pd.read_csv(kan_path).sample(500, random_state=42),
        numerical_columns,
        categorical_columns)


    print("Arquitectura:", arquitectura)
    print("\n================ P-values for Student's t-test: =================")
    print("\n=== TVAE vs REAL ===")
    student_t_tests(numerical_columns, real_data, tvae_data) 
    print("\n=== KAN vs REAL ===")
    student_t_tests(numerical_columns, real_data, kan_data)

    print("\n================ P-values for Mann-Whitney U test: =================")
    print("\n=== TVAE vs REAL ===")
    mann_whitney_tests(numerical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    mann_whitney_tests(numerical_columns, real_data, kan_data)

    print("\n================ P-values for Kolmogorov-Smirnov test: =================")
    print("\n=== TVAE vs REAL ===")
    ks_tests(numerical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    ks_tests(numerical_columns, real_data, kan_data)

    print("\n================ P-values for Chi-squared test: =================")
    print("\n=== TVAE vs REAL ===")     
    chi_squared_tests(categorical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    chi_squared_tests(categorical_columns, real_data, kan_data)

    print("\n================ Cosine distances: =================")
    print("\n=== TVAE vs REAL ===")
    cosine_distances(numerical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    cosine_distances(numerical_columns, real_data, kan_data)

    print("\n================ Jensen-Shannon distances: =================")
    print("\n=== TVAE vs REAL ===")
    js_distances(numerical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    js_distances(numerical_columns, real_data, kan_data)

    print("\n================ Wasserstein distances: =================")
    print("\n=== TVAE vs REAL ===")
    wass_distances(numerical_columns, real_data, tvae_data)
    print("\n=== KAN vs REAL ===")
    wass_distances(numerical_columns, real_data, kan_data)

