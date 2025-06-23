import os
import pandas as pd

input_folder = "tiempo_procesado_formula"
output_file = "media_tiempos_symbolic_total.csv"

resumen = []

for file in os.listdir(input_folder):
    if file.endswith(".csv") and file.startswith("tiempos_symbolic_"):
        file_path = os.path.join(input_folder, file)

        df = pd.read_csv(file_path)
        
        if "tiempo" not in df.columns or "modelo" not in df.columns or "arquitectura" not in df.columns:
            print(f"Saltando archivo mal formado: {file}")
            continue

        df = df[df["tiempo"] != 0]

        if df.empty:
            continue

        modelo = df["modelo"].iloc[0].upper()
        arquitectura = f"({df['arquitectura'].iloc[0]})"

        media = df["tiempo"].mean()

        resumen.append({
            "Modelo": modelo,
            "Arquitectura": arquitectura,
            "Mean_Time_seconds": round(media, 6)
        })

final_df = pd.DataFrame(resumen)
final_df.to_csv(output_file, index=False)
