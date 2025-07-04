import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy, ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
from scipy.stats import entropy, ks_2samp, spearmanr
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

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
    def __init__(self, real_data_path, tvae_data_path, kan_data_path, categorical_columns):
        self.real_data = pd.read_csv(real_data_path)
        self.real_data= self.real_data.dropna()
        self.tvae_data = pd.read_csv(tvae_data_path)
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
        self.tvae_data[self.numerical_columns] = self.scaler.transform(self.tvae_data[self.numerical_columns])
        self.kan_data[self.numerical_columns] = self.scaler.transform(self.kan_data[self.numerical_columns])

        # Codificación de datos categóricos
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.real_data[col] = le.fit_transform(self.real_data[col].astype(str))
            self.tvae_data[col] = le.transform(self.tvae_data[col].astype(str))
            self.kan_data[col] = le.transform(self.kan_data[col].astype(str))
            self.le_dict[col] = le

    def compare_data_statistics(self):
        stats_df = pd.DataFrame()
        for col in self.numerical_columns:
            stats_df[col] = [
                self.real_data[col].mean(), self.tvae_data[col].mean(), self.kan_data[col].mean(),
                self.real_data[col].std(), self.tvae_data[col].std(), self.kan_data[col].std()
            ]
        stats_df.index = ['Real Mean', 'TVAE Mean', 'KAN Mean', 'Real Std', 'TVAE Std', 'KAN Std']
        print("\nComparación Estadística:")
        print(stats_df.T)

    def plot_distributions(self):
        for col in self.numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.kdeplot(self.real_data[col], label="Real Data", fill=True, alpha=0.5)
            sns.kdeplot(self.tvae_data[col], label="TVAE Data", fill=True, alpha=0.5)
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
        y_tvae = self.tvae_data[target_column]
        X_tvae = self.tvae_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
        X_train_tvae, X_test_tvae, y_train_tvae, y_test_tvae = train_test_split(X_tvae, y_tvae, test_size=0.2, random_state=42)
        X_train_kan, X_test_kan, y_train_kan, y_test_kan = train_test_split(X_kan, y_kan, test_size=0.2, random_state=42)

        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        print("\nEvaluación de Modelos de Machine Learning:")
        for name, model in classifiers.items():
            model.fit(X_train_tvae, y_train_tvae)
            preds_tvae = model.predict(X_test_real)
            acc_tvae = accuracy_score(y_test_real, preds_tvae)
            f1_tvae = f1_score(y_test_real, preds_tvae, average='weighted')
            print(f"{name} entrenado en TVAE - Accuracy: {acc_tvae:.4f}, F1: {f1_tvae:.4f}")

            model.fit(X_train_kan, y_train_kan)
            preds_kan = model.predict(X_test_real)
            acc_kan = accuracy_score(y_test_real, preds_kan)
            f1_kan = f1_score(y_test_real, preds_kan, average='weighted')
            print(f"{name} entrenado en KAN - Accuracy: {acc_kan:.4f}, F1: {f1_kan:.4f}")

    def compute_ks_test(self):
        print("\nPrueba de Kolmogorov-Smirnov:")
        for col in self.numerical_columns:
            ks_stat_tvae, p_value_tvae = ks_2samp(self.real_data[col], self.tvae_data[col])
            ks_stat_kan, p_value_kan = ks_2samp(self.real_data[col], self.kan_data[col])
            print(f"{col} - TVAE: KS={ks_stat_tvae:.4f}, p={p_value_tvae:.4f}; KAN: KS={ks_stat_kan:.4f}, p={p_value_kan:.4f}")

    def compute_likelihood_fitness(self, n_components=5):
        print("\n**Métrica de Ajuste de Verosimilitud (Likelihood Fitness) usando GMM con tamaños igualados**")


        n_real = len(self.real_data)
        n_tvae = len(self.tvae_data)
        n_kan = len(self.kan_data)
        n_samples = min(n_real, n_tvae, n_kan)

        tvae_sampled = self.tvae_data[self.numerical_columns].sample(n=n_samples, random_state=42)
        kan_sampled = self.kan_data[self.numerical_columns].sample(n=n_samples, random_state=42)
        real_sampled = self.real_data[self.numerical_columns].sample(n=n_samples, random_state=42)
        '''
        n_real = len(self.real_data)
        tvae_sampled = self.tvae_data[self.numerical_columns].sample(n=n_real, random_state=42)
        kan_sampled = self.kan_data[self.numerical_columns].sample(n=n_real, random_state=42)
        real_sampled = self.real_data[self.numerical_columns]'''

        # GMM entrenado sobre datos reales
        gmm_real = GaussianMixture(n_components=n_components, random_state=42)
        gmm_real.fit(real_sampled)

        Lsyn_tvae = gmm_real.score_samples(tvae_sampled).mean()
        Lsyn_kan = gmm_real.score_samples(kan_sampled).mean()

        # GMM entrenado sobre muestras igualadas de TVAE
        gmm_tvae = GaussianMixture(n_components=n_components, random_state=42)
        gmm_tvae.fit(tvae_sampled)

        # GMM entrenado sobre muestras igualadas de KAN
        gmm_kan = GaussianMixture(n_components=n_components, random_state=42)
        gmm_kan.fit(kan_sampled)

        # Verosimilitud del conjunto real bajo GMM entrenado con cada generador
        Ltest_tvae = gmm_tvae.score_samples(real_sampled).mean()
        Ltest_kan = gmm_kan.score_samples(real_sampled).mean()

        print(f"TVAE - Lsyn: {Lsyn_tvae:.4f}, Ltest: {Ltest_tvae:.4f}")
        print(f"KAN  - Lsyn: {Lsyn_kan:.4f}, Ltest: {Ltest_kan:.4f}")




    def compute_js_divergence(self):
        print("\nDivergencia de Jensen-Shannon en columnas categóricas:")
        for col in self.categorical_columns:
            real = self.real_data[col].values
            tvae = self.tvae_data[col].values
            kan = self.kan_data[col].values

            js_tvae = self._js_divergence(real, tvae)
            js_kan = self._js_divergence(real, kan)

            print(f"{col} - TVAE: {js_tvae:.4f}, KAN: {js_kan:.4f}")

    def _js_divergence(self, real, synthetic, epsilon=1e-10):
        real_dist = np.bincount(real) / len(real) + epsilon
        synthetic_dist = np.bincount(synthetic) / len(synthetic) + epsilon
        min_len = min(len(real_dist), len(synthetic_dist))
        return entropy(real_dist[:min_len], synthetic_dist[:min_len])

    def plot_dimensionality_reduction(self, method='PCA'):
        for name, synthetic in [('TVAE', self.tvae_data), ('KAN', self.kan_data)]:
            combined = np.vstack([self.real_data[self.numerical_columns].values,
                                  synthetic[self.numerical_columns].values])
            labels = np.array([0] * len(self.real_data) + [1] * len(synthetic))

            reducer = PCA(n_components=2) if method == 'PCA' else TSNE(n_components=2, perplexity=30, random_state=42)
            projection = reducer.fit_transform(combined)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=labels, palette=['blue', 'red'], alpha=0.5)
            plt.title(f"{method} Projection: Real vs {name}")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"{method.lower()}_{name}.png")
            plt.savefig(path)
            plt.close()

    def compute_pca_and_tsne_metrics(self):
        print("\nMétricas PCA y t-SNE (Error de Reconstrucción y Divergencia KL):")

        for name, synthetic in [('TVAE', self.tvae_data), ('KAN', self.kan_data)]:
            pca = PCA(n_components=2, random_state=42)
            real_proj = pca.fit_transform(self.real_data[self.numerical_columns])
            synth_proj = pca.transform(synthetic[self.numerical_columns])

            min_len = min(len(real_proj), len(synth_proj))
            recon_error = np.mean((real_proj[:min_len] - synth_proj[:min_len]) ** 2)
            explained_var = pca.explained_variance_ratio_.sum()

            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto', n_iter=1000)
            tsne_real = tsne.fit_transform(self.real_data[self.numerical_columns][:min_len])
            tsne_synth = tsne.fit_transform(synthetic[self.numerical_columns][:min_len])
            kl_div = np.mean(np.abs(tsne_real - tsne_synth))

            print(f"{name} - PCA Var: {explained_var:.4f}, Recon. Error: {recon_error:.4f}, t-SNE KL: {kl_div:.4f}")

    def compute_correlation_metrics(self):
        print("\nMétricas de correlación y correlación de Spearman:")

        for name, synthetic in [('TVAE', self.tvae_data), ('KAN', self.kan_data)]:
            # MAE de las matrices de correlación (Pearson)
            real_corr = self.real_data[self.numerical_columns].corr()
            synth_corr = synthetic[self.numerical_columns].corr()
            mae = np.mean(np.abs(real_corr - synth_corr).values)

            # Muestreo para cálculo de correlación de Spearman
            real_sample = self.real_data[self.numerical_columns].sample(frac=1, random_state=42)
            synth_sample = synthetic[self.numerical_columns].sample(frac=1, random_state=42)
            min_len = min(len(real_sample), len(synth_sample))

            spearman_vals = {}
            pvals = {}
            for col in self.numerical_columns:
                coef, pval = spearmanr(real_sample[col][:min_len], synth_sample[col][:min_len])
                spearman_vals[col] = coef
                pvals[col] = pval

            # Contar variables con correlaciones significativas
            significant = sum(1 for p in pvals.values() if p < 0.05)

            print(f"{name} - MAE Corr: {mae:.4f}")
            print(f"{name} - {significant} de {len(self.numerical_columns)} variables con p < 0.05")

            print(f"{name} - Spearman por variable (coef, p-valor):")
            for col in self.numerical_columns:
                print(f"   {col}: coef={spearman_vals[col]:.4f}, p={pvals[col]:.4e}")

            # Guardar gráfico de barras
            plt.figure(figsize=(10, 6))
            plt.bar(spearman_vals.keys(), spearman_vals.values())
            plt.xticks(rotation=45)
            plt.ylabel("Coef. de Spearman")
            plt.title(f"Correlación de Spearman: Real vs {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"spearman_{name}.png"))
            plt.close()


    def compute_frechet_distance(self):
        print("\nDistancia de Frechet entre representaciones reales y sintéticas:")

        def fdist(X, Y):
            mu1, sigma1 = np.mean(X, axis=0), np.cov(X, rowvar=False)
            mu2, sigma2 = np.mean(Y, axis=0), np.cov(Y, rowvar=False)
            covmean = sqrtm(sigma1 @ sigma2)
            dist = np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return dist

        real = self.real_data[self.numerical_columns].values
        tvae = self.tvae_data[self.numerical_columns].values
        kan = self.kan_data[self.numerical_columns].values

        dist_tvae = fdist(real, tvae)
        dist_kan = fdist(real, kan)

        print(f"TVAE - Frechet Distance: {dist_tvae:.4f}")
        print(f"KAN  - Frechet Distance: {dist_kan:.4f}")

    def check_duplicate_rows(self):
        print("\nComprobación de duplicados exactos en datos sintéticos:")

        def count_dupes(synth):
            duplicates = synth[synth.isin(self.real_data.to_dict(orient='list')).all(axis=1)]
            return len(duplicates), len(duplicates) / len(synth)

        dupes_tvae, ratio_tvae = count_dupes(self.tvae_data)
        dupes_kan, ratio_kan = count_dupes(self.kan_data)

        print(f"TVAE - Duplicados: {dupes_tvae}, Ratio: {ratio_tvae:.2%}")
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
            X_tvae = self.tvae_data.drop(columns=[col])
            y_tvae = self.tvae_data[col]
            X_kan = self.kan_data.drop(columns=[col])
            y_kan = self.kan_data[col]

            if is_categorical:
                model_tvae = RandomForestClassifier(n_estimators=100, random_state=42)
                model_kan = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model_tvae = RandomForestRegressor(n_estimators=100, random_state=42)
                model_kan = RandomForestRegressor(n_estimators=100, random_state=42)

            model_tvae.fit(X_tvae, y_tvae)
            model_kan.fit(X_kan, y_kan)

            preds_tvae = model_tvae.predict(X_real)
            preds_kan = model_kan.predict(X_real)

            if is_categorical:
                acc_tvae = accuracy_score(y_real, preds_tvae)
                acc_kan = accuracy_score(y_real, preds_kan)
                print(f"TVAE → Real  - Accuracy: {acc_tvae:.4f}")
                print(f"KAN  → Real  - Accuracy: {acc_kan:.4f}")
            else:
                mae_tvae = mean_absolute_error(y_real, preds_tvae)
                mae_kan = mean_absolute_error(y_real, preds_kan)
                print(f"TVAE → Real  - MAE: {mae_tvae:.4f}")
                print(f"KAN  → Real  - MAE: {mae_kan:.4f}")

        self.plot_dependency_modeling_results()


    def plot_dependency_modeling_results(self):
        maes_tvae = []
        maes_kan = []
        labels = []

        for col in self.real_data.columns:
            if col == target:
                continue  # Excluir esta variable
            if col not in self.numerical_columns and col not in self.categorical_columns:
                continue

            X_real = self.real_data.drop(columns=[col])
            y_real = self.real_data[col]
            X_tvae = self.tvae_data.drop(columns=[col])
            y_tvae = self.tvae_data[col]
            X_kan = self.kan_data.drop(columns=[col])
            y_kan = self.kan_data[col]

            if col in self.categorical_columns:
                model_tvae = RandomForestClassifier(n_estimators=100, random_state=42)
                model_kan = RandomForestClassifier(n_estimators=100, random_state=42)
                model_tvae.fit(X_tvae, y_tvae)
                model_kan.fit(X_kan, y_kan)
                acc_tvae = accuracy_score(y_real, model_tvae.predict(X_real))
                acc_kan = accuracy_score(y_real, model_kan.predict(X_real))
                maes_tvae.append(acc_tvae)
                maes_kan.append(acc_kan)
            else:
                model_tvae = RandomForestRegressor(n_estimators=100, random_state=42)
                model_kan = RandomForestRegressor(n_estimators=100, random_state=42)
                model_tvae.fit(X_tvae, y_tvae)
                model_kan.fit(X_kan, y_kan)
                mae_tvae = mean_absolute_error(y_real, model_tvae.predict(X_real))
                mae_kan = mean_absolute_error(y_real, model_kan.predict(X_real))
                maes_tvae.append(mae_tvae)
                maes_kan.append(mae_kan)

            labels.append(col)

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width / 2, maes_tvae, width, label='TVAE')
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
        print("\nEvaluación TSTR (Train Synthetic, Test Real) y TRTS (Train Real, Test Synthetic):")

        # Separar atributos y clases
        X_real = self.real_data.drop(columns=[target_column])
        y_real = self.real_data[target_column]
        X_tvae = self.tvae_data.drop(columns=[target_column])
        y_tvae = self.tvae_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # TSTR
        model.fit(X_tvae, y_tvae)
        acc_tstr_tvae = accuracy_score(y_real, model.predict(X_real))

        model.fit(X_kan, y_kan)
        acc_tstr_kan = accuracy_score(y_real, model.predict(X_real))

        # TRTS
        model.fit(X_real, y_real)
        acc_trts_tvae = accuracy_score(y_tvae, model.predict(X_tvae))
        acc_trts_kan = accuracy_score(y_kan, model.predict(X_kan))

        print(f"TSTR - TVAE → Real: Accuracy = {acc_tstr_tvae:.4f}")
        print(f"TSTR - KAN  → Real: Accuracy = {acc_tstr_kan:.4f}")
        print(f"TRTS - Real → TVAE: Accuracy = {acc_trts_tvae:.4f}")
        print(f"TRTS - Real → KAN : Accuracy = {acc_trts_kan:.4f}")


    def compare_feature_importances(self, target_column):
        print("\nComparación de Importancia de Variables (Random Forest):")

        X_real = self.real_data.drop(columns=[target_column])
        y_real = self.real_data[target_column]
        X_tvae = self.tvae_data.drop(columns=[target_column])
        y_tvae = self.tvae_data[target_column]
        X_kan = self.kan_data.drop(columns=[target_column])
        y_kan = self.kan_data[target_column]

        rf_real = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_real, y_real)
        rf_tvae = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tvae, y_tvae)
        rf_kan = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_kan, y_kan)

        importances = pd.DataFrame({
            "Variable": X_real.columns,
            "Real": rf_real.feature_importances_,
            "TVAE": rf_tvae.feature_importances_,
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
        tvae = self.tvae_data[self.numerical_columns].sample(n=len(real), random_state=42).values
        kan = self.kan_data[self.numerical_columns].sample(n=len(real), random_state=42).values

        def mmd(x, y):
            xx = rbf_kernel(x, x, gamma)
            yy = rbf_kernel(y, y, gamma)
            xy = rbf_kernel(x, y, gamma)
            return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

        mmd_tvae = mmd(real, tvae)
        mmd_kan = mmd(real, kan)

        print(f"TVAE - MMD: {mmd_tvae:.4f}")
        print(f"KAN  - MMD: {mmd_kan:.4f}")

        # Guardar resultados en archivo de salida
        with open(os.path.join(self.output_dir, "mmd_scores.txt"), "w", encoding="utf-8") as f:
            f.write("Distancia MMD (Maximum Mean Discrepancy)\n")
            f.write(f"TVAE - MMD: {mmd_tvae:.6f}\n")
            f.write(f"KAN  - MMD: {mmd_kan:.6f}\n")





if __name__ == "__main__":
    os.makedirs("dbeval", exist_ok=True)
    sys.stdout = DualOutput("dbeval/results.txt")


    real_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS/Heart Prediction Quantum DatasetSimple.csv"
    tvae_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS/synthetic_heartdisease_mlp.csv"
    kan_path = "/home/gtav-tft/Desktop/paula/eval/DATASETS/synthetic_heartdisease_MultKAN.csv"

    evaluator = SyntheticDataEvaluator(
        real_data_path=real_path,
        tvae_data_path=tvae_path,
        kan_data_path=kan_path,
        categorical_columns= [
    'HeartDisease',
    'Gender'

]
    )

    print("Columnas del dataset real:")
    print(evaluator.real_data.columns)

    print("\n================ COMPARACIÓN ESTADÍSTICA =================")
    evaluator.compare_data_statistics()

    print("\n================ DISTRIBUCIONES KDE =================")
    evaluator.plot_distributions()

    print("\n================ EFICACIA EN MODELOS DE ML =================")
    evaluator.evaluate_ml_efficacy(target_column=target)

    print("\n================ TEST KOLMOGOROV-SMIRNOV =================")
    evaluator.compute_ks_test()

    print("\n================ LIKELIHOOD FITNESS =================")
    evaluator.compute_likelihood_fitness()

    print("\n================ JENSEN-SHANNON DIVERGENCE =================")
    evaluator.compute_js_divergence()

    #print("\n================ PCA Y t-SNE: PROYECCIONES =================")
    #evaluator.plot_dimensionality_reduction(method='PCA')
    #evaluator.plot_dimensionality_reduction(method='t-SNE')

    print("\n================ PCA Y t-SNE: MÉTRICAS =================")
    #evaluator.compute_pca_and_tsne_metrics()

    print("\n================ CORRELACIÓN (MAE & SPEARMAN) =================")
    evaluator.compute_correlation_metrics()

    print("\n================ DISTANCIA DE FRECHET =================")
    #evaluator.compute_frechet_distance()

    print("\n================ COMPROBACIÓN DE DUPLICADOS =================")
    evaluator.check_duplicate_rows()

    print("\n================ RELACIONES ENTRE VARIABLES =================")
    evaluator.evaluate_variable_dependency_modeling()

    print("\n================ TSTR / TRTS =================")
    evaluator.evaluate_tstr_trts(target_column=target)

    print("\n================ COMPARACIÓN DE IMPORTANCIA DE VARIABLES =================")
    evaluator.compare_feature_importances(target_column=target)

    print("\n================ DISTANCIA MMD =================")
    evaluator.compute_mmd()



