import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import math

result_files = {
    "64-32": "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/64-32/dbeval/results_stats.txt",
    "32-16": "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/32-16/dbeval/results_stats.txt",
    "24-12": "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/24-12/dbeval/results_stats.txt",
    "10-5": "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/10-5/dbeval/results_stats.txt",
}


COLORES_MODELO = {
    "ctgan": "#87B6D6",  # Azul
    "KAN":  "#F1B97C",  # Naranja
    }

images_file =  "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/img_compare"

def parse_metrics(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {}

    def extract_metrics(test_name, section_title, value_label):
        # patrón más robusto
        pattern = re.compile(
            rf"================ {re.escape(section_title)}: =================\n\n"
            r"=== ctgan vs REAL ===\n(.*?)\n\n=== KAN vs REAL ===\n(.*?)(?=\n=+|\Z)",
            re.DOTALL
        )
        match = pattern.search(content)
        if not match:
            return

        for model, block in zip(["ctgan", "KAN"], match.groups()):
            rows = re.findall(rf"Column (\w+): {re.escape(value_label)} = ([0-9.]+|inf)", block)
            for col, val in rows:
                val = float("inf") if val == "inf" else float(val)
                metrics[f"{test_name}_{model}_{col}"] = val

    # Pruebas estadísticas
    extract_metrics("TTEST", "P-values for Student's t-test", "p-value")
    extract_metrics("MWUTEST", "P-values for Mann-Whitney U test", "p-value")
    extract_metrics("KSTEST", "P-values for Kolmogorov-Smirnov test", "p-value")
    extract_metrics("CHISQ", "P-values for Chi-squared test", "p-value")

    # Distancias
    extract_metrics("COS", "Cosine distances", "Cosine distance")
    extract_metrics("JS", "Jensen-Shannon distances", "Jensen-Shannon distance")
    extract_metrics("WASS", "Wasserstein distances", "Wasserstein distance")

    return metrics


def comparar_resultados(result_files):
    all_results = {}
    for config, path in result_files.items():
        if os.path.exists(path):
            all_results[config] = parse_metrics(path)
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.insert(0, "Architecture", df.index)  
    output_csv = "./COMP_TAMAÑOSRED/tabla_resultados_comparativos_stats.csv"
    df.to_csv(output_csv, index=True, float_format="%.4f")
    return df
    


def plot_pvalues_ttest(df, images_file):    

    pvalue_pattern = re.compile(r"TTEST_ctgan_(.+)")
    variables = sorted({
        pvalue_pattern.match(col).group(1) for col in df.columns
        if pvalue_pattern.match(col) and f"TTEST_KAN_{pvalue_pattern.match(col).group(1)}" in df.columns
    })
    print("Variables encontradas:", variables)
    data = []
    for idx, row in df.iterrows():
        for var in variables:
            col_ctgan = f"TTEST_ctgan_{var}"
            col_kan  = f"TTEST_KAN_{var}"
            if col_ctgan in row and col_kan in row:
                data.append({"Arquitectura": str(idx), "Variable": var, "Modelo": "ctgan", "Pvalue": row[col_ctgan], "log-Pvalue": -np.log10(row[col_ctgan] if row[col_ctgan] > 1e-10 else 1e-10)})
                data.append({"Arquitectura": str(idx), "Variable": var, "Modelo": "KAN",  "Pvalue": row[col_kan], "log-Pvalue": -np.log10(row[col_kan] if row[col_kan] > 1e-10 else 1e-10)})

    df_plot = pd.DataFrame(data)
    print(df_plot)

    g = sns.catplot(
        data=df_plot, kind="bar",
        x="Variable", y="Pvalue", hue="Modelo", col="Arquitectura",
        col_wrap=2,
        palette=COLORES_MODELO, height=4, aspect=1.4, sharey=True
    )

    g.set_titles("Arquitectura: {col_name}")
    g.set_axis_labels("Variable", "Pvalue")
    for ax in g.axes.flat:
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if labels and any(label != '' for label in labels):
            ax.set_xticklabels(labels, rotation=45, ha="right")


    plt.tight_layout()
    plt.savefig(os.path.join(images_file, "Pvalue_Ttest.png"), dpi=300)
    plt.close()

    



if __name__ == "__main__":
    df = comparar_resultados(result_files)
    plot_pvalues_ttest(df, images_file)
