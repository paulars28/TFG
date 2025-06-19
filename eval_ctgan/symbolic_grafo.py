
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
from sympy import preorder_traversal
import seaborn as sns

symbol_dict = {f'x_{i}': sp.Symbol(f'x_{i}') for i in range(1, 21)}
symbol_dict.update({
    'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
    'sqrt': sp.sqrt, 'log': sp.log, 'tanh': sp.tanh,
    'sigmoid': lambda x: 1 / (1 + sp.exp(-x))
})

def extraer_exprs(ruta_txt):
    with open(ruta_txt, 'r') as f:
        contenido = f.read()
    listas = re.findall(r"\[.*?\]", contenido, re.DOTALL)
    exprs = eval(listas[0], {}, symbol_dict)
    vars_ = eval(listas[1], {}, symbol_dict)
    return exprs, vars_

def extraer_ocurrencias_variable(expr, var):
    ocurrencias = []
    for subexpr in preorder_traversal(expr):
        if isinstance(subexpr, sp.Basic):
            if var in subexpr.free_symbols:
                ocurrencias.append(subexpr)
    return ocurrencias

def tabla_camino_funcional(expr, vars_x):
    filas = []
    for var in vars_x:
        ocurrencias = extraer_ocurrencias_variable(expr, var)
        for subexpr in ocurrencias:
            filas.append({
                "variable": str(var),
                "subexpresión": str(subexpr),
                "tipo_simbólico": type(subexpr).__name__
            })
    if not filas:
        print(f"No se encontraron variables en la expresión: {expr}")
        return pd.DataFrame(columns=["variable", "subexpresión", "tipo_simbólico"])
    return pd.DataFrame(filas).drop_duplicates().sort_values(by="variable")

def guardar_tabla_funcional(df, ruta):
    df.to_csv(ruta, index=False)
    print(f"Tabla guardada en: {ruta}")

def procesar_bloque(exprs, vars_input, nombre_bloque, ruta):
    for j, expr in enumerate(exprs):
        tabla = tabla_camino_funcional(expr, vars_input)
        guardar_tabla_funcional(tabla, f"{ruta}tabla_{nombre_bloque}_{j+1}.csv")

def seguimiento_variable_fuente(variable_fuente, generator_exprs, generator_input_vars, ruta):
    print(f"Generando trazabilidad para {variable_fuente}")
    df_traza_filtrada = resumen_trazabilidad(generator_exprs, generator_input_vars)
    df_traza_filtrada = df_traza_filtrada[df_traza_filtrada["z_j"].str.contains(variable_fuente)]
    df_traza_filtrada.to_csv(f"{ruta}trazabilidad_generator_{variable_fuente}.csv", index=False)
    print(f"Resumen trazabilidad de {variable_fuente} exportado.")

def profundidad_arbol(expr):
    if not isinstance(expr, sp.Basic) or not expr.args:
        return 1
    return 1 + max(profundidad_arbol(arg) for arg in expr.args)

def medir_complejidad(exprs):
    resultados = []
    for i, expr in enumerate(exprs):
        if not isinstance(expr, sp.Basic):
            continue
        funciones_no_lineales = [f for f in [sp.sin, sp.cos, sp.exp, sp.log, sp.tanh] if expr.has(f)]
        resultados.append({
            "índice": i + 1,
            "profundidad": expr.count_ops(),
            "altura_arbol": profundidad_arbol(expr),
            "n_subexpresiones": len(list(preorder_traversal(expr))),
            "funciones_no_lineales": [f.__name__ for f in funciones_no_lineales]
        })
    return pd.DataFrame(resultados)

def matriz_dependencia(exprs, input_vars):
    matriz = []
    for i, expr in enumerate(exprs):
        fila = []
        for var in input_vars:
            count = len(extraer_ocurrencias_variable(expr, var))
            fila.append(count)
        matriz.append(fila)
    df = pd.DataFrame(matriz, columns=[str(v) for v in input_vars])
    df.index = [f"x̂_{i+1}" for i in range(len(exprs))]
    return df

def resumen_trazabilidad(exprs, input_vars):
    trazas = []
    for i, expr in enumerate(exprs):
        for var in input_vars:
            ocurrencias = extraer_ocurrencias_variable(expr, var)
            if ocurrencias:
                funciones = set(type(s).__name__ for s in ocurrencias)
                trazas.append({
                    "x̂_k": f"x̂_{i+1}",
                    "z_j": str(var),
                    "n_ocurrencias": len(ocurrencias),
                    "funciones": ", ".join(funciones)
                })
    return pd.DataFrame(trazas)

arquitectura = "10-5"
generator_path = f"/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/{arquitectura}/modelo_transformersimple/symbolic_formula_generator.txt"
ruta = f"/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/{arquitectura}/modelo_transformersimple/"

generator_exprs, generator_input_vars = extraer_exprs(generator_path)
"""
procesar_bloque(generator_exprs, generator_input_vars, "generator", ruta)
"""
df_comp_generator = medir_complejidad(generator_exprs)
df_comp_generator.to_csv(f"{ruta}complejidad_generator.csv", index=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_comp_generator.sort_values("profundidad", ascending=False), x="índice", y="profundidad", palette="Blues_d")
plt.title("Complejidad por profundidad - Generator")
plt.xlabel("x̂_i")
plt.ylabel("Operaciones simbólicas")
plt.tight_layout()
plt.savefig(f"{ruta}grafico_complejidad_generator.png")
plt.show()

df_dep = matriz_dependencia(generator_exprs, generator_input_vars)
df_dep.to_csv(f"{ruta}matriz_dependencias_generator.csv")
plt.figure(figsize=(10, 6))
sns.heatmap(df_dep, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matriz de dependencias simbólicas: $z_j \\rightarrow \\hat{x}_k$", fontsize=14, fontweight='bold')
plt.xlabel(r"$z_j$", fontsize=12)
plt.ylabel(r"$\hat{x}_k$", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{ruta}matriz_dependencia_decoder.png", dpi=300)
plt.show()

df_traza = resumen_trazabilidad(generator_exprs, generator_input_vars)
df_traza.to_csv(f"{ruta}trazabilidad_generator.csv", index=False)

seguimiento_variable_fuente("x_5", generator_exprs, generator_input_vars, ruta)

