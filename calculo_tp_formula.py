import sympy as sp
import numpy as np
import time
import ast
from sympy import cos, sin, exp, symbols, sympify
from numpy import cos, sin, exp


ruta_decoder = "/home/gtav-tft/Desktop/paula/symbolic_formulas/ctgan_symbolic_formulas/24-12/symbolic_formula_generator_24_12.txt"

import time
import sympy as sp
import numpy as np
import pandas as pd

def split_expressions(expr_line):
    exprs = []
    current = ""
    depth = 0
    for char in expr_line:
        if char == ',' and depth == 0:
            exprs.append(current)
            current = ""
        else:
            current += char
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
    if current:
        exprs.append(current)
    return exprs

def cargar_formulas_simbólicas(path, start_tag):
    with open(path, 'r') as file:
        lines = file.readlines()

    start_index = next(i for i, line in enumerate(lines) if start_tag in line)
    expr_line = lines[start_index + 1].strip()

    if expr_line.startswith('[') and expr_line.endswith(']'):
        expr_line = expr_line[1:-1]

    exprs_str = split_expressions(expr_line)
    exprs = [sp.sympify(expr.strip()) for expr in exprs_str]
    return exprs


decoder_exprs = cargar_formulas_simbólicas(ruta_decoder, "generator SYMBOLIC FORMULA:")

# Definir símbolos
n_variables = max([int(str(sym)[2:]) for expr in decoder_exprs for sym in expr.free_symbols]) + 1
x = sp.symbols(f'x_0:{n_variables}')

# Lambdificar
decoder_funcs = [sp.lambdify(x, expr, modules='numpy') for expr in decoder_exprs]
entrada = [1] * n_variables

# Medir tiempo para cada fórmula por separado durante 1000 ejecuciones
resultados = []
for idx, f in enumerate(decoder_funcs):
    for _ in range(1000):
        t_ini = time.perf_counter()
        _ = f(*entrada)
        t_fin = time.perf_counter()
        resultados.append({
            'symbolic_formula': idx,
            'modelo': 'CTGAN',
            'arquitectura': '24-12',
            'tiempo': t_fin - t_ini
        })

# Guardar en CSV
df = pd.DataFrame(resultados)
df.to_csv('tiempos_symbolic_generator_ctgan_24-12.csv', index=False)


