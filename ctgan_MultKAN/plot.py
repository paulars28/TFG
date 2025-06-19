import torch
import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  
sys.path.insert(0, root_dir)

from kan.MultKAN import MultKAN  

figures_folder = "./plot"

arquitectura = "10-5"
estructura = "generator"  # "generator" o "discriminator"
#modelo_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_state_modelo.pt"
#modelo_cache = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_cache_data_modelo.pt"
modelo_path = "/home/gtav-tft/Desktop/paula/ctgan_MultKAN/MODELOS/final_"+estructura+"_state_modelo.pt"
modelo_cache = "/home/gtav-tft/Desktop/paula/ctgan_MultKAN/MODELOS/final_"+estructura+"_cache_data_modelo.pt"
#modelo_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/resultado_poda_node_ctgan/24-12-0.20generator0.08discriminator/"+estructura+"_pruned_state"
#modelo_cache = "/home/gtav-tft/Desktop/paula/eval_ctgan/resultado_poda_node_ctgan/24-12-0.20generator0.08discriminator/"+estructura+"_pruned_cache_data"
""
device= 'cuda'

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

    x = x_numerical 
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



import matplotlib.pyplot as plt

print("Generando imágenes finales del modelo...")
multkan.plot(folder=figures_folder)
name = estructura + "_plot.png"
plt.savefig(os.path.join(figures_folder, name), dpi=300, bbox_inches='tight')

print(f"Plot final guardado con éxito en {figures_folder}")
