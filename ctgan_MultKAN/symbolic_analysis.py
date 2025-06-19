import sympy as sp
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt

symbol_dict = {f'x_{i}': sp.Symbol(f'x_{i}') for i in range(1, 21)} # definimos las funciones
symbol_dict.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'sqrt': sp.sqrt})

def extraer_exprs_y_vars(ruta_txt):
    with open(ruta_txt, 'r') as f:
        contenido = f.read()

    listas = re.findall(r"\[.*?\]", contenido, re.DOTALL)
    exprs = eval(listas[0], {}, symbol_dict)
    vars_ = eval(listas[1], {}, symbol_dict)
    return exprs, vars_


def extraer_formula_por_latente(generator_path, output_enc_txt_path=None):
    generator_exprs, _ = extraer_exprs_y_vars(generator_path)
    z_symbols = [sp.Symbol(f'z_{i+1}') for i in range(len(generator_exprs))]
    df = pd.DataFrame({
        "z_i": [str(z) for z in z_symbols],
        "z_i = f_i(x)": [sp.pretty(expr, use_unicode=False) for expr in generator_exprs]
    })
    if output_enc_txt_path:
        with open(output_enc_txt_path, "w") as f:
            for i, expr in enumerate(generator_exprs):
                f.write(f"z_{i+1}(x) = {sp.sstr(expr)}\n")



def extraer_formula_por_reconstruccion(discriminator_path, output_dec_txt_path=None):
    discriminator_exprs, _ = extraer_exprs_y_vars(discriminator_path)
    x_symbols = [sp.Symbol(f'y_{i+1}') for i in range(len(discriminator_exprs))]
    df = pd.DataFrame({
        "x̂_i": [str(x) for x in x_symbols],
        "x̂_i = g_i(z)": [sp.pretty(expr, use_unicode=False) for expr in discriminator_exprs]
    })
    if output_dec_txt_path:
        with open(output_dec_txt_path, "w") as f:
            for i, expr in enumerate(discriminator_exprs):
                f.write(f"x̂_{i+1}(z) = {sp.sstr(expr)}\n")


def componer_generator_discriminator(generator_path, discriminator_path, output_path_composed=None):
    
    generator_exprs, generator_vars = extraer_exprs_y_vars(generator_path)
    discriminator_exprs, discriminator_vars = extraer_exprs_y_vars(discriminator_path)

    generator_expressions = [sp.simplify(expr) for expr in generator_exprs]
    discriminator_exprsessions = [sp.simplify(expr) for expr in discriminator_exprs]
    generator_variables = [sp.simplify(var) for var in generator_vars]
    discriminator_variables = [sp.simplify(var) for var in discriminator_vars]

    subs_map = {str(var): generator_expressions[i] for i, var in enumerate(discriminator_variables)}
    composed_exprs = [expr(subs_map) for expr in discriminator_exprsessions]
    
    with open(output_path_composed, "w") as f:
        for i, expr in enumerate(composed_exprs):
            f.write(f"f̂_{i+1}(x) = {sp.sstr(expr)}\n")

    return composed_exprs, generator_vars

def componer_generator_discriminator(generator_path, discriminator_path, output_path_composed=None):
    print("Composición de expresiones generator y discriminator para obtener la función compuesta definitiva!")
    generator_exprs, generator_vars = extraer_exprs_y_vars(generator_path)
    discriminator_exprs, discriminator_vars = extraer_exprs_y_vars(discriminator_path)

    subs_map = {str(var): generator_exprs[i] for i, var in enumerate(discriminator_vars)}
    composed_exprs = [expr.subs(subs_map) for expr in discriminator_exprs]
    with open("/home/gtav-tft/Desktop/paula/eval/COMP_TAMA\u00d1OSRED/"+arquitectura+"/symbolic_formula_composed.txt", "w") as f:
        for i, expr in enumerate(composed_exprs):
            f.write(f"f̂_{i+1}(x) = {sp.sstr(expr)}\n")

    return composed_exprs, generator_vars

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

    if arquitectura == "generator":
        columnas = [f'x_{i+1}' for i in range(n_columnas)]
        indices = [f'z_{i+1}' for i in range(n_filas)]
    elif arquitectura == "discriminator":
        columnas = [f'z_{i+1}' for i in range(n_columnas)]
        indices = [f'ẋ_{i+1}' for i in range(n_filas)]

    df = pd.DataFrame(matriz, columns=columnas, index=indices)
    df.to_csv(ruta_csv)
    print(f"[csv guardado en: {ruta_csv}]")

def plot_heatmap_contribuciones(matriz, arquitectura, variables, exprs, ruta_png=None):
    matriz_np = np.array(matriz, dtype=float)
    plt.figure(figsize=(10, 6))

    if arquitectura == "generator":
        plt.imshow(matriz_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Derivada parcial')
        plt.xticks(ticks=range(len(variables)), labels=[f'x_{i+1}' for i in range(len(variables))], rotation=45)
        plt.yticks(ticks=range(len(exprs)), labels=[f'z_{j+1}' for j in range(len(exprs))])
        plt.title("Contribución parcial de variables al generator")
        plt.xlabel("Variables de entrada")
        plt.ylabel("Componentes latentes")

    elif arquitectura == "discriminator":
        plt.imshow(matriz_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Derivada parcial')
        plt.xticks(ticks=range(len(variables)), labels=[f'z_{j+1}' for j in range(len(variables))], rotation=45)
        plt.yticks(ticks=range(len(exprs)), labels=[f'ẋ_{i+1}' for i in range(len(exprs))])
        plt.title("Contribución parcial de variables al discriminator")
        plt.xlabel("Variables latentes")
        plt.ylabel("Variables reconstruidas")

    plt.tight_layout()
    if ruta_png:
        plt.savefig(ruta_png)
        print(f"Heatmap guardado en: {ruta_png}")
    plt.show()

def calcular_contribucion_completa(matriz_generator, matriz_discriminator, generator_vars, discriminator_vars, ruta_csv=None):
    A = np.array(matriz_generator)
    B = np.array(matriz_discriminator)
    C = np.dot(B, A)
    df = pd.DataFrame(C, columns=[str(v) for v in generator_vars],
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

arquitectura = "10-5"
 
if __name__ == "__main__":

    generator_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/symbolic_formula_generator.txt"
    #discriminator_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/symbolic_formula_discriminator.txt"
    output_gen_txt_path="/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/symbolic_formula_generator_z.txt"
    #output_disc_txt_path="/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/symbolic_formula_discriminator_x.txt"
    output_path_composed = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/symbolic_formula_composed.txt"

    #composed_exprs, input_vars = componer_generator_discriminator(generator_path, discriminator_path)
    df_generator_z = extraer_formula_por_latente(generator_path, output_gen_txt_path)
    #f_discriminator_x = extraer_formula_por_reconstruccion(discriminator_path, output_dec_txt_path)

    generator_exprs, generator_input_vars = extraer_exprs_y_vars(generator_path)
    matriz_generator = calcular_matriz_contribuciones(generator_exprs, generator_input_vars)

    guardar_contribuciones_csv(matriz_generator,generator_input_vars, generator_exprs,"/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/" + arquitectura + "/modelo_transformersimple/contribuciones_generator_" + arquitectura + ".csv","generator")
    plot_heatmap_contribuciones(matriz_generator,"generator",generator_input_vars,generator_exprs,"/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/" + arquitectura + "/modelo_transformersimple/contribuciones_generator_" + arquitectura + ".png" )

        
  










