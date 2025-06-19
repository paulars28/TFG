import sympy as sp
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

symbol_dict = {f'x_{i}': sp.Symbol(f'x_{i}') for i in range(1, 33)} 
symbol_dict.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'sqrt': sp.sqrt, 'log': sp.log, 'tanh': sp.tanh, 'sigmoid': lambda x: 1 / (1 + sp.exp(-x))})


arquitectura = "64-32" 
import sympy as sp
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Diccionario de símbolos extendido
symbol_dict = {f'x_{i}': sp.Symbol(f'z_{i}') for i in range(1, 33)}  # Interpretar x_i como z_i
symbol_dict.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'sqrt': sp.sqrt, 'log': sp.log, 'tanh': sp.tanh, 'sigmoid': lambda x: 1 / (1 + sp.exp(-x))})

arquitectura = "64-32"

def extraer_exprs_y_vars(ruta_txt):
    with open(ruta_txt, 'r') as f:
        contenido = f.read()

    listas = re.findall(r"\[.*?\]", contenido, re.DOTALL)
    exprs = eval(listas[0], {}, symbol_dict)
    vars_ = eval(listas[1], {}, symbol_dict)
    print(f"{len(exprs)} expresiones y {len(vars_)} variables.")
    return exprs, vars_

def extraer_formula_por_latente(encoder_path, output_enc_txt_path=None):
    print("Extrayendo expresiones del encoder")
    encoder_exprs, _ = extraer_exprs_y_vars(encoder_path)
    z_symbols = [sp.Symbol(f'z_{i+1}') for i in range(len(encoder_exprs))]
    df = pd.DataFrame({
        "z_i": [str(z) for z in z_symbols],
        "z_i = f_i(x)": [sp.pretty(expr, use_unicode=False) for expr in encoder_exprs]
    })
    if output_enc_txt_path:
        with open(output_enc_txt_path, "w") as f:
            for i, expr in enumerate(encoder_exprs):
                f.write(f"z_{i+1}(x) = {sp.sstr(expr)}\n")

def extraer_formula_por_reconstruccion(decoder_path, output_dec_txt_path=None):
    print("Extrayendo expresiones del decoder")
    decoder_exprs, decoder_vars = extraer_exprs_y_vars(decoder_path)

    # Usar z_1, z_2, ... en lugar de x_1, x_2, ...
    z_symbols = [sp.Symbol(f'z_{i+1}') for i in range(len(decoder_vars))]
    sustituciones = dict(zip(decoder_vars, z_symbols))
    decoder_exprs_z = [expr.subs(sustituciones) for expr in decoder_exprs]
    x_symbols = [sp.Symbol(f'y_{i+1}') for i in range(len(decoder_exprs))]
    df = pd.DataFrame({
        "x̂_i": [str(x) for x in x_symbols],
        "x̂_i = g_i(z)": [sp.pretty(expr, use_unicode=False) for expr in decoder_exprs_z] })
    if output_dec_txt_path:
        with open(output_dec_txt_path, "w") as f:
            for i, expr in enumerate(decoder_exprs_z):
                f.write(f"x̂_{i+1}(z) = {sp.sstr(expr)}\n")
    return df

def componer_encoder_decoder(encoder_path, decoder_path, output_path_composed=None):
    print("Composición de expresiones encoder y decoder para obtener la función compuesta definitiva!")
    encoder_exprs, encoder_vars = extraer_exprs_y_vars(encoder_path)
    decoder_exprs, decoder_vars = extraer_exprs_y_vars(decoder_path)

    subs_map = {str(var): encoder_exprs[i] for i, var in enumerate(decoder_vars)}
    composed_exprs = [expr.subs(subs_map) for expr in decoder_exprs]
    with open("/home/gtav-tft/Desktop/paula/eval/COMP_TAMA\u00d1OSRED/"+arquitectura+"/symbolic_formula_composed.txt", "w") as f:
        for i, expr in enumerate(composed_exprs):
            f.write(f"f̂_{i+1}(x) = {sp.sstr(expr)}\n")

    return composed_exprs, encoder_vars

def calcular_matriz_contribuciones(exprs, variables, punto_base=None):
    if punto_base is None:
        punto_base = {v: 0.0 for v in variables}
    matriz = []
    for j, expr in enumerate(exprs):
        fila = []
        for var in variables:
            try:
                derivada = expr.diff(var)
                contrib = abs(derivada.evalf(subs=punto_base))
            except AttributeError:
                contrib = 0.0
            fila.append(contrib)
        matriz.append(fila)
    return matriz

def guardar_contribuciones_csv(matriz, variables, exprs, ruta_csv, arquitectura):
    n_filas = len(matriz)
    n_columnas = len(matriz[0]) if matriz else 0

    if arquitectura == "encoder":
        columnas = [f'x_{i+1}' for i in range(n_columnas)]
        indices = [f'z_{i+1}' for i in range(n_filas)]
    elif arquitectura == "decoder":
        columnas = [f'z_{i+1}' for i in range(n_columnas)]
        indices = [f'ẋ_{i+1}' for i in range(n_filas)]

    df = pd.DataFrame(matriz, columns=columnas, index=indices)
    df.to_csv(ruta_csv)
    print(f"[csv guardado en: {ruta_csv}]")

def plot_heatmap_contribuciones(matriz, arquitectura, variables, exprs, ruta_png=None):
    matriz_np = np.array(matriz, dtype=float)
    plt.figure(figsize=(10, 6))

    if arquitectura == "encoder":
        plt.imshow(matriz_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Derivada parcial')
        plt.xticks(ticks=range(len(variables)), labels=[f'x_{i+1}' for i in range(len(variables))], rotation=45)
        plt.yticks(ticks=range(len(exprs)), labels=[f'z_{j+1}' for j in range(len(exprs))])
        plt.title("Contribución parcial de variables al encoder")
        plt.xlabel("Variables de entrada")
        plt.ylabel("Componentes latentes")

    elif arquitectura == "decoder":
        plt.imshow(matriz_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Derivada parcial')
        plt.xticks(ticks=range(len(variables)), labels=[f'z_{j+1}' for j in range(len(variables))], rotation=45)
        plt.yticks(ticks=range(len(exprs)), labels=[f'ẋ_{i+1}' for i in range(len(exprs))])
        plt.title("Contribución parcial de variables al decoder")
        plt.xlabel("Variables latentes")
        plt.ylabel("Variables reconstruidas")

    plt.tight_layout()
    if ruta_png:
        plt.savefig(ruta_png)
        print(f"Heatmap guardado en: {ruta_png}")
    plt.show()

def calcular_contribucion_completa(matriz_encoder, matriz_decoder, encoder_vars, decoder_vars, ruta_csv=None):
    A = np.array(matriz_encoder)
    B = np.array(matriz_decoder)
    C = np.dot(B, A)
    df = pd.DataFrame(C, columns=[str(v) for v in encoder_vars],
                         index=[f"x̂_{k+1}" for k in range(len(C))])
    if ruta_csv:
        df.to_csv(ruta_csv)
        print(f"[OK] Matriz de contribución compuesta guardada en: {ruta_csv}")
    return df

def plot_heatmap_contribucion_total(df_contrib, ruta_png=None):
    matriz_np = df_contrib.values.astype(float)
    plt.figure(figsize=(10, 6))
    plt.imshow(matriz_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Contribución total')
    plt.xticks(ticks=range(len(df_contrib.columns)), labels=df_contrib.columns, rotation=45)
    plt.yticks(ticks=range(len(df_contrib.index)), labels=df_contrib.index)
    plt.title("Influencia total de variables de entrada sobre reconstrucción")
    plt.xlabel("Variables de entrada $x_i$")
    plt.ylabel("Variables reconstruidas $\hat{x}_k$")
    plt.tight_layout()
    if ruta_png:
        plt.savefig(ruta_png)
        print(f"Heatmap guardado en: {ruta_png}")
    plt.show()




 
if __name__ == "__main__":

    encoder_path = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_encoder.txt"
    decoder_path = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_decoder.txt"
    output_enc_txt_path="/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_encoder_z.txt"
    output_dec_txt_path="/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_decoder_x.txt"
    output_path_composed = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_composed.txt"
    ruta_csv_compuesta = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribuciones_compuestas_"+arquitectura+".csv"
    #composed_exprs, input_vars = componer_encoder_decoder(encoder_path, decoder_path)
    #df_encoder_z = extraer_formula_por_latente(encoder_path, output_enc_txt_path)
    #df_decoder_x = extraer_formula_por_reconstruccion(decoder_path, output_dec_txt_path)
    encoder_exprs, encoder_input_vars = extraer_exprs_y_vars(encoder_path)
    decoder_exprs, decoder_input_vars = extraer_exprs_y_vars(decoder_path)
    matriz_encoder = calcular_matriz_contribuciones(encoder_exprs, encoder_input_vars)
    matriz_decoder = calcular_matriz_contribuciones(decoder_exprs, decoder_input_vars)
    guardar_contribuciones_csv(matriz_encoder, encoder_input_vars, encoder_exprs, "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribuciones_encoder_"+arquitectura+".csv", "encoder")
    guardar_contribuciones_csv(matriz_decoder, decoder_input_vars, decoder_exprs, "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribuciones_decoder_"+arquitectura+".csv", "decoder")
    plot_heatmap_contribuciones(matriz_encoder, "encoder", encoder_input_vars, encoder_exprs, "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribuciones_encoder_"+arquitectura+".png")
    plot_heatmap_contribuciones(matriz_decoder, "decoder", decoder_input_vars, decoder_exprs, "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribuciones_decoder_"+arquitectura+".png")
    df_contribucion_total = calcular_contribucion_completa(matriz_encoder, matriz_decoder, encoder_input_vars, decoder_input_vars, ruta_csv_compuesta)
    plot_heatmap_contribucion_total(df_contribucion_total, "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/eval/symbolic/contribucion_total_"+arquitectura+".png")

    
  










