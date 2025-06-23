import os
import pandas as pd

input_folder = "tiempo_procesado"
output_file = "media_tiempos_procesamiento_total.csv"

resumen = []

for file in os.listdir(input_folder):
    if file.endswith(".csv") and file.startswith("samples_"):
        file_path = os.path.join(input_folder, file)

        parts = file.replace("samples_", "").replace(".csv", "").split("_")
        if len(parts) != 4:
            continue

        model = parts[0].upper()
        arch_str = parts[1].replace("-", ", ")
        architecture = f"({arch_str})"
        device = parts[2].upper()
        model_type = parts[3]  

        df = pd.read_csv(file_path)
        df = df[df['Time_seconds'] != 0]  

        media = df['Time_seconds'].mean()

        resumen.append({
            "Model": model,
            "Architecture": architecture,
            "Device": device,
            "Type": model_type,
            "Mean_Time_seconds": round(media, 4)
        })

final_df = pd.DataFrame(resumen)
final_df.to_csv(output_file, index=False)
