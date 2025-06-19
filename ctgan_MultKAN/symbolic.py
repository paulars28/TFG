import sys
import os
import torch
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from kan.MultKAN import MultKAN  

poda = "node" #node/edge
arquitectura = "24-12"
estructura = "generator"
modelo_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/final_"+estructura+"_state_modelo.pt"
modelo_cache = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/modelo_transformersimple/final_"+estructura+"_cache_data_modelo.pt"
device= 'cuda'
state_dict = torch.load(modelo_path, map_location="cpu")

#ver parametros por terminal
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")


def reconstruccion_checkpoint(modelo_path, device='cpu'):
    dic = torch.load(modelo_path, device)
    width = []
    i = 0
    while f'act_fun.{i}.coef' in dic:
        in_dim, out_dim, n_splines = dic[f'act_fun.{i}.coef'].shape
        if i == 0:
            width.append(in_dim)
        width.append(out_dim)
        i += 1


    print(f"Reconstruyendo MultKAN con dimensiones: {width}")
    multkan = MultKAN(width=width, device=device)
    multkan.load_state_dict(dic, strict=True)
    return multkan


from synthesizers.ctgan import Discriminator 
from synthesizers.ctgan import Generator

compress_dims = [10, 5] 
embedding_dim = 5

if estructura == "discriminator":
    modelo = Discriminator(input_dim=embedding_dim, discriminator_dim=compress_dims, pac=10).to(device)
elif estructura == "generator":
    modelo = Generator(embedding_dim=embedding_dim, generator_dim=compress_dims, data_dim=16).to(device)
else:
    raise ValueError("estructura debe ser 'generator' o 'discriminator'")

multkan = reconstruccion_checkpoint(modelo_path, device=device)
multkan.cache_data = torch.load(modelo_cache, map_location=device)
multkan.eval()


print("Establecemos activaciones antes de prune_node...")
acts = []
acts_premult = []
spline_preacts = []
spline_postacts = []
spline_postsplines = []
acts_scale = []
acts_scale_spline = []
subnode_actscale = []
edge_actscale = []


if estructura == "generator":
    modelo.generator_multkan = multkan
    net = modelo.generator_multkan
elif estructura == "discriminator":
    modelo.discriminator_multkan = multkan
    net = modelo.discriminator_multkan
else:
    raise ValueError("estructura no válida")

x = net.cache_data.to(device)

acts.append(x)

for l in range(net.depth):
    x_numerical, preacts, postacts_numerical, postspline = net.act_fun[l](x)

    if net.symbolic_enabled:
        x_symbolic, _ = net.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000)
    else:
        x_symbolic = 0.

    x = x_numerical + x_symbolic
    subnode_actscale.append(torch.std(x, dim=0).detach())
    x = net.subnode_scale[l][None, :] * x + net.subnode_bias[l][None, :]

    input_range = torch.std(preacts, dim=0) + 0.1
    output_range = torch.std(postacts_numerical, dim=0)
    edge_actscale.append(output_range)
    acts_scale.append((output_range / input_range).detach())
    acts_scale_spline.append(output_range / input_range)
    spline_preacts.append(preacts.detach())
    spline_postacts.append(postacts_numerical.detach())
    spline_postsplines.append(postspline.detach())

    acts_premult.append(x.detach())
    acts.append(x.detach())


# Asignamos las activaciones restauradas
net.acts = acts
net.acts_premult = acts_premult
net.spline_preacts = spline_preacts
net.spline_postacts = spline_postacts
net.spline_postsplines = spline_postsplines
net.acts_scale = acts_scale
net.acts_scale_spline = acts_scale_spline
net.subnode_actscale = subnode_actscale
net.edge_actscale = edge_actscale

net.attribute()



# Extraer y guardar fórmula simbólica
try:
    if estructura == "generator":
        modelo.generator_multkan.auto_symbolic()
        symbolic_formula = modelo.generator_multkan.symbolic_formula()

    elif estructura == "discriminator":
        modelo.discriminator_multkan.auto_symbolic()
        symbolic_formula = modelo.discriminator_multkan.symbolic_formula()

    with open("symbolic_formula_" + estructura + ".txt", "w") as f:
        f.write(estructura + " SYMBOLIC FORMULA:\n")
        f.write("\n".join([str(eq) for eq in symbolic_formula]))
    print("Fórmula simbólica final guardada en symbolic_formula.txt")

except Exception as e:
    print(f"Fallo al extraer la función simbólica: {e}")



