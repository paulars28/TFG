import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

def load_eval_arrays(model_name, input_dir="./eval_arrays"):

    path = os.path.join(input_dir, model_name + "_eval.npz")
    data = np.load(path)
    return data["labels"], data["probs"]



def compare_models(metric_files, save_csv=False, output_file="comparativa_modelos.csv"):

    metric_pattern = {
        "Accuracy": r"Accuracy:\s+([0-9.]+)",
        "Precision": r"Precision:\s+([0-9.]+)",
        "Recall": r"Recall:\s+([0-9.]+)",
        "F1 Score": r"F1 Score:\s+([0-9.]+)",
        "ROC AUC": r"ROC AUC:\s+([0-9.]+)",
        "PR AUC": r"PR AUC:\s+([0-9.]+)",
        "Matthews CorrCoef": r"Matthews CorrCoef:\s+([0-9.]+)"
    }

    data = []

    for file in metric_files:
        model_name = os.path.basename(file).replace("_metricas.txt", "")
        with open(file, "r") as f:
            content = f.read()

        row = {"Model": model_name}
        for key, pattern in metric_pattern.items():
            match = re.search(pattern, content)
            row[key] = float(match.group(1)) if match else None

        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("Model")

    if save_csv:
        df.to_csv(output_file)
        print(f" Tabla comparativa guardada como '{output_file}'")

    return df

def obtener_etiquetas_y_probabilidades(modelo, ruta_modelo, test_loader, dispositivo):

    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    modelo.to(dispositivo)
    modelo.eval()

    todas_labels = []
    todas_probs = []

    with torch.no_grad():
        for entradas, etiquetas in test_loader:
            entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
            salidas = modelo(entradas)
            probs = torch.sigmoid(salidas).squeeze()

            todas_probs.extend(probs.cpu().numpy())
            todas_labels.extend(etiquetas.cpu().numpy())

    return np.array(todas_labels), np.array(todas_probs)

COLORES_MODELO = {
    "KAN": "#F1B97C",  # Naranja intermedio 
    "MLP": "#87B6D6"   # Azul intermedio 
}

COLORES_MODELO_osc = {
    "KAN": "#E07B39",  # Naranja pastel oscuro
    "MLP": "#6BAED6"   # Azul pastel oscuro
}

def grafica_comparacion_modelos(df, metricas=None, tipo="bar", guardar=False, archivo_salida="comparativa_modelos.png"):
    if metricas is None:
        metricas = df.columns.tolist()

    if tipo == "bar":
        df_plot = df[metricas].T
        colores = [COLORES_MODELO.get(col, "gray") for col in df_plot.columns]
        df_plot.plot(kind="bar", figsize=(12, 6), color=colores)
        plt.title("Comparativa de modelos por métrica", fontsize=14)
        plt.ylabel("Valor", fontsize=12)
        plt.xlabel("Métrica", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, fontsize=10)
        plt.legend(title="Modelo", fontsize=10)
        plt.tight_layout()

    elif tipo == "radar":
        num_vars = len(metricas)
        angulos = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angulos += angulos[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, fila in df.iterrows():
            valores = fila[metricas].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, label=i, color=COLORES_MODELO.get(i, 'gray'))
            ax.fill(angulos, valores, alpha=0.1, color=COLORES_MODELO.get(i, 'gray'))

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angulos[:-1]), metricas)
        ax.set_title("Radar de métricas por modelo", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=10)

    else:
        raise ValueError("Tipo de gráfico no soportado. Usa 'bar' o 'radar'.")

    if guardar:
        plt.savefig(archivo_salida, dpi=300)
        print(f"Gráfico guardado como '{archivo_salida}'")

    plt.show()

def comparacion_curvas_evaluacion(diccionario_labels_probs, nombres_modelos=("KAN", "MLP"), guardar=False, prefijo="comparativa"):
    plt.figure()
    for modelo in nombres_modelos:
        etiquetas, probs = diccionario_labels_probs[modelo]
        fpr, tpr, _ = roc_curve(etiquetas, probs)
        auc_val = roc_auc_score(etiquetas, probs)
        plt.plot(fpr, tpr, label=f"{modelo} (AUC = {auc_val:.4f})", color=COLORES_MODELO_osc.get(modelo, 'gray'), linewidth=1.25)

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1.25)
    plt.xlabel("Tasa de falsos positivos", fontsize=12)
    plt.ylabel("Tasa de verdaderos positivos", fontsize=12)
    plt.title("Comparativa de curvas ROC", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if guardar:
        plt.savefig(f"{prefijo}_roc_comparativa.png", dpi=300)
    plt.show()

    plt.figure()
    for modelo in nombres_modelos:
        etiquetas, probs = diccionario_labels_probs[modelo]
        prec, rec, _ = precision_recall_curve(etiquetas, probs)
        pr_auc = average_precision_score(etiquetas, probs)
        plt.plot(rec, prec, label=f"{modelo} (AP = {pr_auc:.4f})", color=COLORES_MODELO_osc.get(modelo, 'gray'), linewidth=1.25)

    plt.xlabel("Recall (Sensibilidad)", fontsize=12)
    plt.ylabel("Precisión", fontsize=12)
    plt.title("Comparativa de curvas Precisión-Recall", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if guardar:
        plt.savefig(f"{prefijo}_pr_comparativa.png", dpi=300)
        print(f"Comparativas guardadas como '{prefijo}_roc_comparativa.png' y '{prefijo}_pr_comparativa.png'")
    plt.show()

if __name__ == "__main__":

    files = [
        "MLP_metricas.txt",
        "KAN_metricas.txt",]
    df_comparativa = compare_models(files, save_csv=True)
    grafica_comparacion_modelos(df_comparativa, tipo="bar", guardar=True, archivo_salida="comparativa_modelos.png")

    all_labels_kan, all_probs_kan = load_eval_arrays("KAN")
    all_labels_mlp, all_probs_mlp = load_eval_arrays("MLP")
    comparacion_curvas_evaluacion(
        diccionario_labels_probs={
            "KAN": (all_labels_kan, all_probs_kan),
            "MLP": (all_labels_mlp, all_probs_mlp)},
        nombres_modelos=("KAN", "MLP"),
        guardar=True,
        prefijo="kan_vs_mlp")


