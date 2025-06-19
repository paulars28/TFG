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

target = "HeartDisease"  


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


class SyntheticDataEvaluator:
    def __init__(self, real_data_path, ctgan_data_path, kan_data_path, categorical_columns):
        self.real_data = pd.read_csv(real_data_path)
        self.real_data= self.real_data.dropna()
        self.ctgan_data = pd.read_csv(ctgan_data_path)
        self.kan_data = pd.read_csv(kan_data_path)
        self.categorical_columns = categorical_columns
        self.numerical_columns = self.real_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.scaler = MinMaxScaler()
        self.le_dict = {}
        self.output_dir = "dbeval"
        os.makedirs(self.output_dir, exist_ok=True)
        self._preprocess_data()

    def _preprocess_data(self):
        # Escalado de datos numéricos
        self.real_data[self.numerical_columns] = self.scaler.fit_transform(self.real_data[self.numerical_columns])
        self.ctgan_data[self.numerical_columns] = self.scaler.transform(self.ctgan_data[self.numerical_columns])
        self.kan_data[self.numerical_columns] = self.scaler.transform(self.kan_data[self.numerical_columns])
        # Codificación de datos categóricos
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.real_data[col] = le.fit_transform(self.real_data[col].astype(str))
            self.ctgan_data[col] = le.transform(self.ctgan_data[col].astype(str))
            self.kan_data[col] = le.transform(self.kan_data[col].astype(str))
            self.le_dict[col] = le


    def compare_data_statistics(self):
        stats_df = pd.DataFrame()
        for col in self.numerical_columns:
            stats_df[col] = [
                self.real_data[col].mean(), self.ctgan_data[col].mean(), self.kan_data[col].mean(),
                self.real_data[col].std(), self.ctgan_data[col].std(), self.kan_data[col].std()
            ]
        stats_df.index = ['Real Mean', 'ctgan Mean', 'KAN Mean', 'Real Std', 'ctgan Std', 'KAN Std']
        print("\nComparación Estadística:")
        print(stats_df.T)


    def plot_distributions(self):
        for col in self.numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.kdeplot(self.real_data[col], label="Real Data", fill=True, alpha=0.5)
            sns.kdeplot(self.ctgan_data[col], label="ctgan Data", fill=True, alpha=0.5)
            sns.kdeplot(self.kan_data[col], label="KAN Data", fill=True, alpha=0.5)
            plt.legend()
            plt.title(f"Comparación de Distribuciones para {col}")
            plt.tight_layout()
            fig_path = os.path.join(self.output_dir, f"dist_{col}.png")
            plt.savefig(fig_path)
            plt.close()


    def evaluate_ml_efficacy(self, target_column):
        y_real = self.real_data[target_column]
        X_real = self.real_data.drop(columns=[target_column])
        y_ctgan = self.ctgan_data[target_column]
        X_ctgan = self.ctgan_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
        X_train_ctgan, X_test_ctgan, y_train_ctgan, y_test_ctgan = train_test_split(X_ctgan, y_ctgan, test_size=0.2, random_state=42)
        X_train_kan, X_test_kan, y_train_kan, y_test_kan = train_test_split(X_kan, y_kan, test_size=0.2, random_state=42)
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        print("\nEvaluación de Modelos de Machine Learning:")
        for name, model in classifiers.items():
            model.fit(X_train_ctgan, y_train_ctgan)
            preds_ctgan = model.predict(X_test_real)
            acc_ctgan = accuracy_score(y_test_real, preds_ctgan)
            f1_ctgan = f1_score(y_test_real, preds_ctgan, average='weighted')
            print(f"{name} entrenado en ctgan - Accuracy: {acc_ctgan:.4f}, F1: {f1_ctgan:.4f}")
            model.fit(X_train_kan, y_train_kan)
            preds_kan = model.predict(X_test_real)
            acc_kan = accuracy_score(y_test_real, preds_kan)
            f1_kan = f1_score(y_test_real, preds_kan, average='weighted')
            print(f"{name} entrenado en KAN - Accuracy: {acc_kan:.4f}, F1: {f1_kan:.4f}")


    def compute_ks_test(self):
        print("\nPrueba de Kolmogorov-Smirnov:")
        print("{:<20} {:>10} {:>10} {:>10} {:>10}".format("Variable", "KS-ctgan", "p-ctgan", "KS-KAN", "p-KAN"))
        print("-" * 60)
        for col in self.numerical_columns:
            ks_stat_ctgan, p_value_ctgan = ks_2samp(self.real_data[col], self.ctgan_data[col])
            ks_stat_kan, p_value_kan = ks_2samp(self.real_data[col], self.kan_data[col])
            print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                col, ks_stat_ctgan, p_value_ctgan, ks_stat_kan, p_value_kan
            ))


    def compute_likelihood_fitness(self, n_components=5):
        print("\n**Métrica de Ajuste de Verosimilitud (Likelihood Fitness) usando GMM con tamaños igualados**")
        n_real = len(self.real_data)
        n_ctgan = len(self.ctgan_data)
        n_kan = len(self.kan_data)
        n_samples = min(n_real, n_ctgan, n_kan)
        ctgan_sampled = self.ctgan_data[self.numerical_columns].sample(n=n_samples, random_state=42)
        kan_sampled = self.kan_data[self.numerical_columns].sample(n=n_samples, random_state=42)
        real_sampled = self.real_data[self.numerical_columns].sample(n=n_samples, random_state=42)

        # GMM entrenado sobre datos reales
        gmm_real = GaussianMixture(n_components=n_components, random_state=42)
        gmm_real.fit(real_sampled)
        Lsyn_ctgan = gmm_real.score_samples(ctgan_sampled).mean()
        Lsyn_kan = gmm_real.score_samples(kan_sampled).mean()

        # GMM entrenado sobre muestras igualadas de ctgan
        gmm_ctgan = GaussianMixture(n_components=n_components, random_state=42)
        gmm_ctgan.fit(ctgan_sampled)

        # GMM entrenado sobre muestras igualadas de KAN
        gmm_kan = GaussianMixture(n_components=n_components, random_state=42)
        gmm_kan.fit(kan_sampled)

        # Verosimilitud del conjunto real bajo GMM entrenado con cada generador
        Ltest_ctgan = gmm_ctgan.score_samples(real_sampled).mean()
        Ltest_kan = gmm_kan.score_samples(real_sampled).mean()
        print(f"ctgan - Lsyn: {Lsyn_ctgan:.4f}, Ltest: {Ltest_ctgan:.4f}")
        print(f"KAN  - Lsyn: {Lsyn_kan:.4f}, Ltest: {Ltest_kan:.4f}")



    def compute_correlation_metrics(self):
        print("\nMétricas de correlación y correlación de Spearman:")

        for name, synthetic in [('ctgan', self.ctgan_data), ('KAN', self.kan_data)]:
            print(f"\n=== {name} ===")

            # MAE de matrices de correlación (Pearson)
            real_corr = self.real_data[self.numerical_columns].corr()
            synth_corr = synthetic[self.numerical_columns].corr()
            mae = np.mean(np.abs(real_corr - synth_corr).values)
            print(f"MAE de la matriz de correlación (Pearson): {mae:.4f}")

            # Muestreo para Spearman
            real_sample = self.real_data[self.numerical_columns].sample(frac=1, random_state=42)
            synth_sample = synthetic[self.numerical_columns].sample(frac=1, random_state=42)
            min_len = min(len(real_sample), len(synth_sample))

            # Cálculo de Spearman por variable
            spearman_vals = {}
            pvals = {}
            for col in self.numerical_columns:
                coef, pval = spearmanr(real_sample[col][:min_len], synth_sample[col][:min_len])
                spearman_vals[col] = coef
                pvals[col] = pval

            # Conteo de correlaciones significativas
            significant = sum(1 for p in pvals.values() if p < 0.05)
            print(f"Variables con correlación de Spearman significativa (p < 0.05): {significant} / {len(self.numerical_columns)}")

            # Tabla Spearman
            print("{:<20} {:>10} {:>12}".format("Variable", "Coef", "p-valor"))
            print("-" * 45)
            for col in self.numerical_columns:
                print("{:<20} {:>10.4f} {:>12.4e}".format(col, spearman_vals[col], pvals[col]))



    def compute_frechet_distance(self):
        print("\nDistancia de Frechet entre representaciones reales y sintéticas:")
        def fdist(X, Y):
            mu1, sigma1 = np.mean(X, axis=0), np.cov(X, rowvar=False)
            mu2, sigma2 = np.mean(Y, axis=0), np.cov(Y, rowvar=False)
            covmean = sqrtm(sigma1 @ sigma2)
            dist = np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return dist
        real = self.real_data[self.numerical_columns].values
        ctgan = self.ctgan_data[self.numerical_columns].values
        kan = self.kan_data[self.numerical_columns].values

        dist_ctgan = fdist(real, ctgan)
        dist_kan = fdist(real, kan)

        print(f"ctgan - Frechet Distance: {dist_ctgan:.4f}")
        print(f"KAN  - Frechet Distance: {dist_kan:.4f}")




    def check_duplicate_rows(self):
        print("\nComprobación de duplicados exactos en datos sintéticos:")
        def count_dupes(synth):
            duplicates = synth[synth.isin(self.real_data.to_dict(orient='list')).all(axis=1)]
            return len(duplicates), len(duplicates) / len(synth)

        dupes_ctgan, ratio_ctgan = count_dupes(self.ctgan_data)
        dupes_kan, ratio_kan = count_dupes(self.kan_data)

        print(f"ctgan - Duplicados: {dupes_ctgan}, Ratio: {ratio_ctgan:.2%}")
        print(f"KAN  - Duplicados: {dupes_kan}, Ratio: {ratio_kan:.2%}")



    def evaluate_variable_dependency_modeling(self):
        print("\nEvaluación de preservación de relaciones entre variables (Modelado supervisado multivariable):")
        for col in self.real_data.columns:
            if col == target:
                continue  # Excluir esta variable
            if col not in self.numerical_columns and col not in self.categorical_columns:
                continue

            print(f"\nVariable objetivo: {col}")
            is_categorical = col in self.categorical_columns

            X_real = self.real_data.drop(columns=[col])
            y_real = self.real_data[col]
            X_ctgan = self.ctgan_data.drop(columns=[col])
            y_ctgan = self.ctgan_data[col]
            X_kan = self.kan_data.drop(columns=[col])
            y_kan = self.kan_data[col]

            if is_categorical:
                model_ctgan = RandomForestClassifier(n_estimators=100, random_state=42)
                model_kan = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model_ctgan = RandomForestRegressor(n_estimators=100, random_state=42)
                model_kan = RandomForestRegressor(n_estimators=100, random_state=42)

            model_ctgan.fit(X_ctgan, y_ctgan)
            model_kan.fit(X_kan, y_kan)

            preds_ctgan = model_ctgan.predict(X_real)
            preds_kan = model_kan.predict(X_real)

            if is_categorical:
                acc_ctgan = accuracy_score(y_real, preds_ctgan)
                acc_kan = accuracy_score(y_real, preds_kan)
                print(f"ctgan → Real  - Accuracy: {acc_ctgan:.4f}")
                print(f"KAN  → Real  - Accuracy: {acc_kan:.4f}")
            else:
                mae_ctgan = mean_absolute_error(y_real, preds_ctgan)
                mae_kan = mean_absolute_error(y_real, preds_kan)
                print(f"ctgan → Real  - MAE: {mae_ctgan:.4f}")
                print(f"KAN  → Real  - MAE: {mae_kan:.4f}")

        self.plot_dependency_modeling_results()



    def plot_dependency_modeling_results(self):
        maes_ctgan = []
        maes_kan = []
        labels = []
        for col in self.real_data.columns:
            if col == target:
                continue  # Excluir esta variable
            if col not in self.numerical_columns and col not in self.categorical_columns:
                continue

            X_real = self.real_data.drop(columns=[col])
            y_real = self.real_data[col]
            X_ctgan = self.ctgan_data.drop(columns=[col])
            y_ctgan = self.ctgan_data[col]
            X_kan = self.kan_data.drop(columns=[col])
            y_kan = self.kan_data[col]

            if col in self.categorical_columns:
                model_ctgan = RandomForestClassifier(n_estimators=100, random_state=42)
                model_kan = RandomForestClassifier(n_estimators=100, random_state=42)
                model_ctgan.fit(X_ctgan, y_ctgan)
                model_kan.fit(X_kan, y_kan)
                acc_ctgan = accuracy_score(y_real, model_ctgan.predict(X_real))
                acc_kan = accuracy_score(y_real, model_kan.predict(X_real))
                maes_ctgan.append(acc_ctgan)
                maes_kan.append(acc_kan)
            else:
                model_ctgan = RandomForestRegressor(n_estimators=100, random_state=42)
                model_kan = RandomForestRegressor(n_estimators=100, random_state=42)
                model_ctgan.fit(X_ctgan, y_ctgan)
                model_kan.fit(X_kan, y_kan)
                mae_ctgan = mean_absolute_error(y_real, model_ctgan.predict(X_real))
                mae_kan = mean_absolute_error(y_real, model_kan.predict(X_real))
                maes_ctgan.append(mae_ctgan)
                maes_kan.append(mae_kan)

            labels.append(col)

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, maes_ctgan, width, label='ctgan')
        bars2 = ax.bar(x + width / 2, maes_kan, width, label='KAN')
        ax.set_ylabel("MAE (o Accuracy si es categórica)")
        ax.set_title("Preservación de relaciones entre variables")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "variable_dependency_comparison.png")
        plt.savefig(output_path)
        plt.close()




    def evaluate_tstr_trts(self, target_column):
        print("\nEvaluación TSTR/TRTS con múltiples modelos y métricas:")

        X_real = self.real_data.drop(columns=[target_column])
        y_real = self.real_data[target_column]
        X_ctgan = self.ctgan_data.drop(columns=[target_column])
        y_ctgan = self.ctgan_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]

        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
        }

        def evaluate_model(model, X_train, y_train, X_test, y_test):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average='binary' if type_of_target(y_test) == 'binary' else 'weighted'),
                    "precision": precision_score(y_test, y_pred, average='binary' if type_of_target(y_test) == 'binary' else 'weighted'),
                    "recall": recall_score(y_test, y_pred, average='binary' if type_of_target(y_test) == 'binary' else 'weighted'),
                }
                # AUC solo para problemas binarios con probas
                if type_of_target(y_test) == 'binary':
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                return metrics

        results = {}

        for clf_name, clf in classifiers.items():
            results[clf_name] = {
                "TSTR_ctgan": evaluate_model(clf, X_ctgan, y_ctgan, X_real, y_real),
                "TSTR_KAN": evaluate_model(clf, X_kan, y_kan, X_real, y_real),
                "TRTS_ctgan": evaluate_model(clf, X_real, y_real, X_ctgan, y_ctgan),
                "TRTS_KAN": evaluate_model(clf, X_real, y_real, X_kan, y_kan),
            }

        # Presentación ordenada
        for clf_name, clf_results in results.items():
            print(f"\n=== {clf_name} ===")
            for eval_type, metrics in clf_results.items():
                metric_str = ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"{eval_type}: {metric_str}")
        
        return results  # Útil para posterior exportación o análisis



    def compare_feature_importances(self, target_column):
        print("\nComparación de Importancia de Variables (Random Forest):")

        X_real = self.real_data.drop(columns=[target_column])
        y_real = self.real_data[target_column]
        X_ctgan = self.ctgan_data.drop(columns=[target_column])
        y_ctgan = self.ctgan_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]
        rf_real = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_real, y_real)
        rf_ctgan = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ctgan, y_ctgan)
        rf_kan = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_kan, y_kan)
        importances = pd.DataFrame({
            "Variable": X_real.columns,
            "Real": rf_real.feature_importances_,
            "ctgan": rf_ctgan.feature_importances_,
            "KAN": rf_kan.feature_importances_,
        })
        print(importances)
        importances.set_index("Variable").plot.bar(figsize=(12, 6))
        plt.title("Importancia de Variables - Random Forest")
        plt.ylabel("Importancia")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(self.output_dir, "feature_importance_comparison.png"))
        plt.close()

        

    def compute_mmd(self, gamma=1.0):
        print("\n**Distancia MMD (Maximum Mean Discrepancy) con kernel RBF**")

        real = self.real_data[self.numerical_columns].values
        ctgan = self.ctgan_data[self.numerical_columns].sample(n=len(real), random_state=42).values
        kan = self.kan_data[self.numerical_columns].sample(n=len(real), random_state=42).values
        def mmd(x, y):
            xx = rbf_kernel(x, x, gamma)
            yy = rbf_kernel(y, y, gamma)
            xy = rbf_kernel(x, y, gamma)
            return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
        mmd_ctgan = mmd(real, ctgan)
        mmd_kan = mmd(real, kan)
        print(f"ctgan - MMD: {mmd_ctgan:.4f}")
        print(f"KAN  - MMD: {mmd_kan:.4f}")






if __name__ == "__main__":
    os.makedirs("dbeval", exist_ok=True)
    sys.stdout = DualOutput("dbeval/results.txt")


    real_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/DATASETS_EVALUATION/Heart Prediction Quantum Dataset.csv"
    ctgan_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/DATASETS_EVALUATION/synthetic_heartdisease_mlp.csv"
    kan_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/DATASETS_EVALUATION/synthetic_heartdisease_kan.csv"

    evaluator = SyntheticDataEvaluator(
        real_data_path=real_path,
        ctgan_data_path=ctgan_path,
        kan_data_path=kan_path,
        categorical_columns= [
        'HeartDisease'])

    categorical_columns= [
    'HeartDisease',
    'Gender']



    print("Columnas del dataset real:")
    print(evaluator.real_data.columns)

    print("\n================ COMPARACIÓN ESTADÍSTICA =================")
    evaluator.compare_data_statistics()

    print("\n================ EFICACIA EN MODELOS DE ML =================")
    evaluator.evaluate_ml_efficacy(target_column=target)

    #print("\n================ TEST KOLMOGOROV-SMIRNOV =================")
    #evaluator.compute_ks_test()

    print("\n================ LIKELIHOOD FITNESS =================")
    evaluator.compute_likelihood_fitness()

    print("\n================ TSTR / TRTS =================")
    evaluator.evaluate_tstr_trts(target_column=target)

    print("\n================ CORRELACIÓN (MAE & SPEARMAN) =================")
    evaluator.compute_correlation_metrics()

    print("\n================ COMPROBACIÓN DE DUPLICADOS =================")
    evaluator.check_duplicate_rows()

    print("\n================ RELACIONES ENTRE VARIABLES =================")
    evaluator.evaluate_variable_dependency_modeling()

    print("\n================ COMPARACIÓN DE IMPORTANCIA DE VARIABLES =================")
    evaluator.compare_feature_importances(target_column=target)

    print("\n================ DISTANCIA MMD =================")
    evaluator.compute_mmd()



