import sympy as sp
import numpy as np
import pandas as pd


import sympy as sp
import numpy as np

arquitectura = "16-8"  
arq = "16_8"  

from tvae_MultKAN.demo import load_demo
from tvae_MultKAN.data_transformer import DataTransformer


real_data = load_demo()


discrete_columns = [
    col for col in real_data.columns
    if real_data[col].dtype == 'object' or real_data[col].nunique() < 20
]

transformer = DataTransformer()
transformer.fit(real_data, discrete_columns)
real_data_transformed = transformer.transform(real_data)

idx = np.random.randint(0, real_data_transformed.shape[0])
x_input_np = real_data_transformed[idx]
x_input = x_input_np.tolist()


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

    try:
        start_index = next(i for i, line in enumerate(lines) if start_tag in line)
    except StopIteration:
        raise ValueError(f"No se encontró el tag '{start_tag}' en el archivo '{path}'.")

    expr_line = lines[start_index + 1].strip()
    if expr_line.startswith('[') and expr_line.endswith(']'):
        expr_line = expr_line[1:-1]

    exprs_str = split_expressions(expr_line)
    exprs = [sp.sympify(expr.strip()) for expr in exprs_str]
    return exprs


ruta_encoder = "/home/gtav-tft/Desktop/paula/symbolic_formulas/tvae_symbolic_formulas/"+arquitectura+"/symbolic_formula_encoder_"+arq+".txt"
ruta_decoder = "/home/gtav-tft/Desktop/paula/symbolic_formulas/tvae_symbolic_formulas/"+arquitectura+"/symbolic_formula_decoder_"+arq+".txt"

encoder_exprs = cargar_formulas_simbólicas(ruta_encoder, "encoder SYMBOLIC FORMULA:")
decoder_exprs = cargar_formulas_simbólicas(ruta_decoder, "decoder SYMBOLIC FORMULA:")


n_input_vars = max([
    int(str(sym)[2:])
    for expr in encoder_exprs
    for sym in expr.free_symbols
    if str(sym).startswith('x_')
])
x_syms = sp.symbols(f'x_1:{n_input_vars + 1}')  
encoder_funcs = [
    sp.lambdify(x_syms, expr, modules=[{
        'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
        'sqrt': np.sqrt, 'Abs': np.abs, 'log': np.log
    }, 'numpy'])
    for expr in encoder_exprs
]

decoder_funcs = []
for expr in decoder_exprs:
    vars_used = sorted(
        [sym for sym in expr.free_symbols if str(sym).startswith('z_')],
        key=lambda s: int(str(s).split('_')[1])
    )
    f = sp.lambdify(vars_used, expr, modules=[{
        'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
        'sqrt': np.sqrt, 'Abs': np.abs, 'log': np.log
    }, 'numpy'])
    decoder_funcs.append((f, vars_used, expr)) 


z = np.array([float(f(*x_input)) for f in encoder_funcs])
z_dict = {f'z_{i+1}': z[i] for i in range(len(z))}

    
from sympy.utilities.lambdify import implemented_function

x_reconstructed = []
for f, vars_used, expr in decoder_funcs:
    try:
        if not vars_used:
            evaluated = expr.evalf()
            if evaluated.free_symbols:
                print("\n--- Expresión constante con símbolos libres ---")
                print("Expr original:", expr)
                print("Expr evaluado:", evaluated)
                print("Símbolos libres:", evaluated.free_symbols)
                raise ValueError(f"La expresión constante aún tiene símbolos libres.")
            val = float(evaluated)
        else:
            args = [float(z_dict[str(sym)]) for sym in vars_used]
            val = float(f(*args))
        x_reconstructed.append(val)
    except Exception as e:
        print("\n--- Error evaluando expresión ---")
        print("Expr original:", expr)
        print("Vars usadas:", vars_used)
        print("Expr tipo:", type(expr))
        raise e

x_input_np = np.array(x_input)
x_reconstructed_np = np.array(x_reconstructed)

error_abs = np.sum(np.abs(x_input_np - x_reconstructed_np))

denominator = np.sum(np.abs(x_input_np))
error_rel = error_abs / denominator if denominator != 0 else np.inf

print("\n[RESULTADOS]")
print("Input x:", np.round(x_input_np, 4).tolist())
print("Reconstrucción x̂:", np.round(x_reconstructed_np, 4).tolist())
print(f"Error absoluto: {error_abs:.4f}")
print(f"Error relativo: {error_rel:.4f}")

print(f"[DEBUG] Longitud x_i: {len(x_input_np)}")
print(f"[DEBUG] Longitud x̂: {len(x_reconstructed_np)}")
print(f"[DEBUG] Longitud fórmulas decoder: {len(decoder_exprs)}")


errores_abs = np.abs(x_input_np - x_reconstructed_np)
errores_rel = errores_abs / (np.abs(x_input_np) + 1e-8)  

df = pd.DataFrame({
    "Input x": np.round(x_input_np, 4),
    "Reconstrucción x̂": np.round(x_reconstructed_np, 4),
    "Error absoluto": np.round(errores_abs, 4),
    "Error relativo": np.round(errores_rel, 4)
})

df.to_csv("simetria_results_"+arquitectura+".csv", index=False)


n_samples = 100
errores = []
for _ in range(n_samples):
    idx = np.random.randint(0, real_data_transformed.shape[0])
    x_input_np = real_data_transformed[idx]
    x_input = x_input_np.tolist()
    z = np.array([float(f(*x_input)) for f in encoder_funcs])
    z_dict = {f'z_{i+1}': z[i] for i in range(len(z))}
    x_reconstructed = []
    for f, vars_used, expr in decoder_funcs:
        if not vars_used:
            val = float(expr.evalf())
        else:
            args = [float(z_dict[str(sym)]) for sym in vars_used]
            val = float(f(*args))
        x_reconstructed.append(val)
    x_input_np = np.array(x_input)
    x_reconstructed_np = np.array(x_reconstructed)
    error_abs = np.sum(np.abs(x_input_np - x_reconstructed_np))
    denom = np.sum(np.abs(x_input_np))
    error_rel = error_abs / denom if denom != 0 else np.inf
    errores.append((error_abs, error_rel))


errores_abs, errores_rel = zip(*errores)
df_summary = pd.DataFrame({
    "Error absoluto": errores_abs,
    "Error relativo": errores_rel
})
df_summary.loc["Media"] = [
    np.mean(errores_abs),
    np.mean(errores_rel)
]
df_summary.to_csv("simetria_summary_"+arquitectura+".csv", index=True)