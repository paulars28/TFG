import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

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




def plot_model_comparison(df, metrics=None, kind="bar", save=False, output_file="comparativa_modelos.png"):

    if metrics is None:
        metrics = df.columns.tolist()

    if kind == "bar":
        df_plot = df[metrics].T
        df_plot.plot(kind="bar", figsize=(12, 6))
        plt.title("Comparativa de modelos por métrica")
        plt.ylabel("Valor")
        plt.xlabel("Métrica")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend(title="Modelo", loc="upper right")
        plt.tight_layout()

    elif kind == "radar":
        

        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cierre del círculo

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, row in df.iterrows():
            values = row[metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, label=i)
            ax.fill(angles, values, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_title("Radar chart de métricas por modelo")
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    else:
        raise ValueError("Tipo de gráfico no soportado. Usa 'bar' o 'radar'.")

    if save:
        plt.savefig(output_file)
        print(f" Gráfico guardado como '{output_file}'")

    plt.show()


def get_labels_and_probs_from_model(model, model_path, test_loader, device):
    """
    Carga pesos del modelo desde .pth y calcula etiquetas y probabilidades sobre el test_loader.

    Retorna:
        all_labels, all_probs (numpy arrays)
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)



def compare_eval_curves(labels_probs_dict, model_names=("KAN", "MLP"), save=False, prefix="comparativa"):

    # --- ROC Curve ---
    plt.figure()
    for model in model_names:
        labels, probs = labels_probs_dict[model]
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f"{model} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparativa ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_roc_comparativa.png")
    plt.show()

    # --- Precision-Recall Curve ---
    plt.figure()
    for model in model_names:
        labels, probs = labels_probs_dict[model]
        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        plt.plot(rec, prec, label=f"{model} (AP = {pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Comparativa Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_pr_comparativa.png")
    plt.show()

    if save:
        print(f"Comparativas guardadas como '{prefix}_roc_comparativa.png' y '{prefix}_pr_comparativa.png'")







if __name__ == "__main__":

    files = [
        "./prueba4/diabetesMLP_metricas.txt",
        "./prueba4/diabetesKAN_metricas.txt",
    ]
    df_comparativa = compare_models(files, save_csv=True)
    plot_model_comparison(df_comparativa, kind="bar", save=True, output_file="comparativa_modelos.png")


    all_labels_kan, all_probs_kan = load_eval_arrays("diabetesKAN")
    all_labels_mlp, all_probs_mlp = load_eval_arrays("diabetesMLP")
    compare_eval_curves(
        labels_probs_dict={
            "KAN": (all_labels_kan, all_probs_kan),
            "MLP": (all_labels_mlp, all_probs_mlp)
        },
        model_names=("KAN", "MLP"),
        save=True,
        prefix="kan_vs_mlp"
    )


